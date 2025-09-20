from abc import abstractmethod, ABC

import torch
from torch import nn, Size, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import MultivariateNormal, Distribution

from cheetah.particles import ParticleBeam
from cheetah.utils.bmadx import bmad_to_cheetah_coords
from .beams import TransverseBeam4D
import numpy as np

class BeamGenerator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self) -> ParticleBeam | TransverseBeam4D:
        pass

class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # query, key, value
        return x + attn_out  # Residual connection

class ParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        model_dim: int = 16,
        n_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 32,
        dropout: float = 0.1,
        output_dim: int = 6,
        output_scale: float = 1e-2, # This is the initial value
    ):
        """
        Transformer-based transformation of particle phase space.
        Automatically centers output to have zero mean in each dimension.
        """
        super(ParticleTransformer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.output_dim = output_dim

        # Embedding layer for each particle
        self.embedding = nn.Linear(input_dim, model_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, dropout=dropout)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.final_projection = nn.Linear(model_dim, output_dim)

        # --- CHANGE START ---
        # Make output_scale a learnable parameter.
        # We learn the log of the scale to ensure it's always positive when exponentiated.
        # self.log_output_scale = nn.Parameter(torch.tensor(np.log(output_scale + 1e-8), dtype=torch.float32))
        self.output_scale = output_scale
        # --- CHANGE END ---

    def _center_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Center coordinates so each dimension has mean = 0.
        Preserves standard deviation and relative particle positions.
        """
        # Calculate mean for each dimension
        mean_coords = coords.mean(dim=0, keepdim=True)  # Shape: [1, output_dim]
        
        # Subtract mean to center around zero
        centered_coords = coords - mean_coords
        
        return centered_coords

    def forward(self, X: Tensor, chunk_size: int = None) -> Tensor:
        """
        Forward pass that returns raw tensor coordinates, centered around zero.
        """
        # Add fake batch dimension if not present
        if X.dim() == 2:
            X = X.unsqueeze(0)  # [1, n_particles, input_dim]
    
        if chunk_size is None:
            output = self._forward_chunked(X).squeeze(0)
        else:
            batch_size, n_particles, _ = X.shape
            outputs = []
        
            for i in range(0, n_particles, chunk_size):
                chunk = X[:, i:i+chunk_size, :]
                out = self._forward_chunked(chunk)
                outputs.append(out)
        
            output = torch.cat(outputs, dim=1).squeeze(0)
        
        # Center the output coordinates
        centered_output = self._center_coordinates(output)
        
        return centered_output
    
    def _forward_chunked(self, X: Tensor) -> Tensor:
        """Internal forward pass through transformer."""
        x = self.embedding(X)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.final_projection(x)
        return x * self.output_scale
    
    def create_beam(self, X: Tensor, energy: Tensor, particle_charges: Tensor = None, chunk_size: int = None):
        """
        Create appropriate beam object based on output dimensions
        """
        transformed_coords = self.forward(X, chunk_size=chunk_size)
        
        if self.output_dim == 4:
            # Return TransverseBeam4D
            return TransverseBeam4D(
                x=transformed_coords[:, 0],
                px=transformed_coords[:, 1],
                y=transformed_coords[:, 2],
                py=transformed_coords[:, 3],
                particle_charges=particle_charges,
                energy=energy
            )
        elif self.output_dim == 6:
            # Convert to cheetah coordinates and return ParticleBeam
            transformed_coords = bmad_to_cheetah_coords(
                transformed_coords, energy, torch.tensor(0.511e6)
            )
            return ParticleBeam(
                *transformed_coords.T,  # Unpack the 6 coordinates
                particle_charges=particle_charges,
            )
        else:
            raise ValueError(f"Unsupported output_dim: {self.output_dim}. Use 4 or 6.")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class NNTransform(torch.nn.Module):
    def __init__(
        self,
        hidden_widths: list[int],
        input_dim: int = 6,
        output_dim: int = 6,
        n_heads: int = 2,
        dropout: float = 0.0,
        activation: torch.nn.Module = torch.nn.Tanh(),
        output_scale: float = 1e-2,
    ):
        """
        Nonparametric transformation using Transformer-like architecture.
        Automatically centers output to have zero mean in each dimension.
        """
        super(NNTransform, self).__init__()

        if not isinstance(hidden_widths, list):
            raise ValueError("hidden_widths must be a list")

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear embedding layer
        self.embed_layer = torch.nn.Linear(input_dim, hidden_widths[0])

        # Build transformer blocks
        layers = []
        in_width = hidden_widths[0]
        for width in hidden_widths[1:]:
            layers.append(
                self._make_transformer_block(in_width, width, n_heads, dropout, activation)
            )
            in_width = width

        self.transformer_blocks = torch.nn.Sequential(*layers)

        # Final projection
        self.final_layer = torch.nn.Linear(hidden_widths[-1], output_dim)

        # Register output scale buffer
        self.register_buffer("output_scale", torch.tensor(output_scale))

    def _make_transformer_block(self, in_dim, out_dim, n_heads, dropout, activation):
        return torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            SelfAttentionBlock(in_dim, n_heads, dropout),
            torch.nn.Linear(in_dim, out_dim),
            activation,
            torch.nn.Dropout(dropout),
        )

    def _center_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Center coordinates so each dimension has mean = 0.
        Preserves standard deviation and relative particle positions.
        """
        # Calculate mean for each dimension
        mean_coords = coords.mean(dim=0, keepdim=True)  # Shape: [1, output_dim]
        
        # Subtract mean to center around zero
        centered_coords = coords - mean_coords
        
        return centered_coords

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns raw tensor coordinates, centered around zero.
        """
        # X shape: [n_particles, input_dim]
        X = X.unsqueeze(1)  # [n_particles, 1, input_dim]
        x = self.embed_layer(X)  # [n_particles, 1, hidden_widths[0]]
        x = self.transformer_blocks(x)  # [n_particles, 1, hidden_widths[-1]]
        x = x.squeeze(1)  # [n_particles, hidden_widths[-1]]
        x = self.final_layer(x)  # [n_particles, output_dim]
        output = x * self.output_scale
        
        # Center the output coordinates
        centered_output = self._center_coordinates(output)
        
        return centered_output
    
    def create_beam(self, X: torch.Tensor, energy: torch.Tensor, particle_charges: torch.Tensor = None):
        """
        Create appropriate beam object based on output dimensions
        """
        transformed_coords = self.forward(X)
        
        if self.output_dim == 4:
            # Return TransverseBeam4D
            return TransverseBeam4D(
                x=transformed_coords[:, 0],
                px=transformed_coords[:, 1],
                y=transformed_coords[:, 2], 
                py=transformed_coords[:, 3],
                particle_charges=particle_charges,
                energy=energy
            )
        elif self.output_dim == 6:
            # Convert to cheetah coordinates and return ParticleBeam
            transformed_coords = bmad_to_cheetah_coords(
                transformed_coords, energy, torch.tensor(0.511e6)
            )
            return ParticleBeam(
                *transformed_coords.T,  # Unpack the 6 coordinates
                particle_charges=particle_charges,
            )
        else:
            raise ValueError(f"Unsupported output_dim: {self.output_dim}. Use 4 or 6.")


class NNParticleBeamGenerator(BeamGenerator):
    def __init__(
        self,
        n_particles: int,
        energy: float,
        base_dist: Distribution = MultivariateNormal(torch.zeros(6), torch.eye(6)),
        transformer: ParticleTransformer | NNTransform = ParticleTransformer(
            input_dim=6,
            model_dim=16,
            n_heads=4,
            num_layers=2,
            dim_feedforward=32,
            dropout=0.1,
            output_dim=6,
            output_scale=1e-2
        ),
    ):
        super(NNParticleBeamGenerator, self).__init__()
        self.transformer = transformer
        self.base_dist = base_dist
        self.register_buffer("beam_energy", torch.tensor(energy))
        self.register_buffer("particle_charges", torch.tensor(1.0))

        self.set_base_particles(n_particles)

    def set_base_particles(self, n_particles: int):
        # Adjust base distribution dimension based on transformer input_dim
        if self.transformer.input_dim == 4:
            if self.base_dist.event_shape[0] == 6:
                # Create 4D base distribution from 6D
                self.base_dist = MultivariateNormal(torch.zeros(4), torch.eye(4))
        elif self.transformer.input_dim == 6:
            if self.base_dist.event_shape[0] == 4:
                # Create 6D base distribution from 4D  
                self.base_dist = MultivariateNormal(torch.zeros(6), torch.eye(6))
                
        self.register_buffer(
            "base_particles", self.base_dist.sample(Size([n_particles]))
        )

    def forward(self):
        """Generate beam using the transformer's create_beam method"""
        if isinstance(self.transformer, ParticleTransformer):
            return self.transformer.create_beam(
                X=self.base_particles,
                energy=self.beam_energy,
                particle_charges=self.particle_charges,
                chunk_size=1000
            )
        else:
            return self.transformer.create_beam(
                X=self.base_particles,
                energy=self.beam_energy,
                particle_charges=self.particle_charges,
            )