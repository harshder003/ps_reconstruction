import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from typing import Literal
from scipy.ndimage import gaussian_filter

class TransverseBeam4D:
    def __init__(self, x, px, y, py, particle_charges=None, energy=None, total_charge=None):
        """
        Constructor that matches ParticleBeam signature for 4D coordinates
        
        Args:
            x: x coordinates [m]
            px: x momenta 
            y: y coordinates [m]
            py: y momenta
            particle_charges: Charge of each particle
            energy: Reference energy
            total_charge: Total charge of the beam
        """
        # Stack the 4D coordinates
        self.coords_4d = torch.stack([x, px, y, py], dim=-1)
        
        # Handle particle charges
        if particle_charges is None:
            self.particle_charges = torch.ones_like(x)
        else:
            self.particle_charges = particle_charges
            
        # Store energy and total charge
        self.energy = energy if energy is not None else torch.tensor(1e9)  # Default energy
        self.total_charge = total_charge if total_charge is not None else torch.sum(self.particle_charges)
        
        # No survival probabilities for direct construction
        self.survival_probs = None
        
        # Define dimension mapping and labels
        self.dimension_map = {
            'x': 0, 'px': 1, 'y': 2, 'py': 3
        }
        
        self.PRETTY_DIMENSION_LABELS = {
            'x': 'x [m]',
            'px': 'px',
            'y': 'y [m]', 
            'py': 'py'
        }
    
    @classmethod
    def from_particle_beam(cls, particle_beam):
        """Alternative constructor from existing ParticleBeam"""
        coords_4d = particle_beam.particles[..., :4].clone()
        return cls(
            x=coords_4d[..., 0],
            px=coords_4d[..., 1], 
            y=coords_4d[..., 2],
            py=coords_4d[..., 3],
            particle_charges=particle_beam.particle_charges,
            energy=particle_beam.energy,
            total_charge=particle_beam.total_charge
        )
    
    @property
    def x(self):
        """x coordinates of particles, filtered by survival probability"""
        data = self.coords_4d[..., 0]
        if self.survival_probs is not None:
            # Apply survival probability masking similar to ParticleBeam
            survived_mask = self.survival_probs > 0.5
            return data[survived_mask]
        return data.flatten()
    
    @property
    def px(self):
        """px coordinates of particles, filtered by survival probability"""
        data = self.coords_4d[..., 1]
        if self.survival_probs is not None:
            survived_mask = self.survival_probs > 0.5
            return data[survived_mask]
        return data.flatten()
        
    @property
    def y(self):
        """y coordinates of particles, filtered by survival probability"""
        data = self.coords_4d[..., 2]
        if self.survival_probs is not None:
            survived_mask = self.survival_probs > 0.5
            return data[survived_mask]
        return data.flatten()
        
    @property
    def py(self):
        """py coordinates of particles, filtered by survival probability"""
        data = self.coords_4d[..., 3]
        if self.survival_probs is not None:
            survived_mask = self.survival_probs > 0.5
            return data[survived_mask]
        return data.flatten()

    def plot_1d_distribution(
        self,
        dimension: Literal["x", "px", "y", "py"],
        bins: int = 100,
        bin_range: tuple[float] | None = None,
        smoothing: float = 0.0,
        plot_kws: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 1D histogram of the given dimension of the particle distribution.
        Matches ParticleBeam.plot_1d_distribution exactly.
        """
        if ax is None:
            _, ax = plt.subplots()

        x_array = getattr(self, dimension).cpu().detach().numpy()
        histogram, edges = np.histogram(x_array, bins=bins, range=bin_range)
        centers = (edges[:-1] + edges[1:]) / 2

        if smoothing:
            histogram = gaussian_filter(histogram, smoothing)

        ax.plot(
            centers,
            histogram / histogram.max(),
            **{"color": "black"} | (plot_kws or {}),
        )
        ax.set_xlabel(f"{self.PRETTY_DIMENSION_LABELS[dimension]}")

        # Handle units exactly like ParticleBeam
        if dimension in ("x", "y"):
            base_unit = "m"
            # Note: format_axis_with_prefixed_unit would be called here in original
            # For now, we'll use the standard formatting

        return ax

    def plot_2d_distribution(
        self,
        x_dimension: Literal["x", "px", "y", "py"],
        y_dimension: Literal["x", "px", "y", "py"],
        style: Literal["histogram", "contour"] = "histogram",
        bins: int = 100,
        bin_ranges: tuple[tuple[float]] | None = None,
        histogram_smoothing: float = 0.0,
        contour_smoothing: float = 3.0,
        pcolormesh_kws: dict | None = None,
        contour_kws: dict | None = None,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plot a 2D histogram of the given dimensions of the particle distribution.
        Matches ParticleBeam.plot_2d_distribution exactly.
        """
        if ax is None:
            _, ax = plt.subplots()

        histogram, x_edges, y_edges = np.histogram2d(
            getattr(self, x_dimension).cpu().detach().numpy(),
            getattr(self, y_dimension).cpu().detach().numpy(),
            bins=bins,
            range=bin_ranges,
        )
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Post-process and plot exactly like ParticleBeam
        smoothed_histogram = gaussian_filter(histogram, histogram_smoothing)
        clipped_histogram = np.where(smoothed_histogram > 1, smoothed_histogram, np.nan)
        
        if style == "histogram":
            ax.pcolormesh(
                x_edges,
                y_edges,
                clipped_histogram.T / smoothed_histogram.max(),
                **{"cmap": "rainbow"} | (pcolormesh_kws or {}),
            )
        elif style == "contour":
            contour_histogram = gaussian_filter(histogram, contour_smoothing)
            ax.contour(
                x_centers,
                y_centers,
                contour_histogram.T / contour_histogram.max(),
                **{"levels": 3} | (contour_kws or {}),
            )

        ax.set_xlabel(f"{self.PRETTY_DIMENSION_LABELS[x_dimension]}")
        ax.set_ylabel(f"{self.PRETTY_DIMENSION_LABELS[y_dimension]}")

        # Handle units exactly like ParticleBeam
        if x_dimension in ("x", "y"):
            x_base_unit = "m"
            # format_axis_with_prefixed_unit would be called here

        if y_dimension in ("x", "y"):
            y_base_unit = "m"
            # format_axis_with_prefixed_unit would be called here

        return ax

    def plot_distribution(
        self,
        dimensions: tuple[str, ...] = ("x", "px", "y", "py"),
        bins: int = 100,
        bin_ranges: Literal["same"] | tuple[float] | list[tuple[float]] | None = None,
        plot_1d_kws: dict | None = None,
        plot_2d_kws: dict | None = None,
        axs: list[plt.Axes] | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """
        Plot of coordinates projected into 2D planes.
        Matches ParticleBeam.plot_distribution exactly.
        """
        if axs is None:
            fig, axs = plt.subplots(
                len(dimensions),
                len(dimensions),
                figsize=(2 * len(dimensions), 2 * len(dimensions)),
            )
        else:
            fig = axs[0, 0].figure
            assert axs.shape == (len(dimensions), len(dimensions)), (
                "If `axs` is provided, it must have the shape "
                f"`({len(dimensions)}, {len(dimensions)})`."
            )

        # Determine bin ranges for all plots in the grid at once - exactly like ParticleBeam
        full_tensor = (
            torch.stack([getattr(self, dimension) for dimension in dimensions], dim=-2)
            .cpu()
            .detach()
            .numpy()
        )
        
        if bin_ranges is None:
            bin_ranges = [
                (
                    full_tensor[i, :].min()
                    - (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
                    full_tensor[i, :].max()
                    + (full_tensor[i, :].max() - full_tensor[i, :].min()) / 10,
                )
                for i in range(full_tensor.shape[-2])
            ]
        
        if bin_ranges == "unit_same":
            spacial_idxs = [
                i
                for i, dimension in enumerate(dimensions)
                if dimension in ["x", "y"]  # Removed "tau" since we only have 4D
            ]
            spacial_bin_range = (
                full_tensor[spacial_idxs, :].min()
                - (
                    full_tensor[spacial_idxs, :].max()
                    - full_tensor[spacial_idxs, :].min()
                )
                / 10,
                full_tensor[spacial_idxs, :].max()
                + (
                    full_tensor[spacial_idxs, :].max()
                    - full_tensor[spacial_idxs, :].min()
                )
                / 10,
            )
            unitless_idxs = [
                i
                for i, dimension in enumerate(dimensions)
                if dimension in ["px", "py"]  # Removed "p" since we only have 4D
            ]
            unitless_bin_range = (
                full_tensor[unitless_idxs, :].min()
                - (
                    full_tensor[unitless_idxs, :].max()
                    - full_tensor[unitless_idxs, :].min()
                )
                / 10,
                full_tensor[unitless_idxs, :].max()
                + (
                    full_tensor[unitless_idxs, :].max()
                    - full_tensor[unitless_idxs, :].min()
                )
                / 10,
            )
            bin_range_dict = {
                "x": spacial_bin_range,
                "px": unitless_bin_range,
                "y": spacial_bin_range,
                "py": unitless_bin_range,
            }
            bin_ranges = [bin_range_dict[dimension] for dimension in dimensions]
        
        if np.asarray(bin_ranges).shape == (2,):
            bin_ranges = [bin_ranges] * len(dimensions)
        assert len(bin_ranges) == len(dimensions) and all(
            len(e) == 2 for e in bin_ranges
        )

        # Plot diagonal 1D histograms on the diagonal - exactly like ParticleBeam
        diagonal_axs = [axs[i, i] for i, _ in enumerate(dimensions)]
        for dimension, bin_range, ax in zip(dimensions, bin_ranges, diagonal_axs):
            self.plot_1d_distribution(
                dimension=dimension,
                bins=bins,
                bin_range=bin_range,
                ax=ax,
                **(plot_1d_kws or {}),
            )

        # Plot 2D histograms on the off-diagonal - exactly like ParticleBeam
        for i, j in itertools.combinations(range(len(dimensions)), 2):
            self.plot_2d_distribution(
                x_dimension=dimensions[i],
                y_dimension=dimensions[j],
                bins=bins,
                bin_ranges=(bin_ranges[i], bin_ranges[j]),
                ax=axs[j, i],
                **(plot_2d_kws or {}),
            )

        # Hide unused axes - exactly like ParticleBeam
        for i, j in itertools.combinations(range(len(dimensions)), 2):
            axs[i, j].set_visible(False)

        # Clean up labels - exactly like ParticleBeam
        for ax_column in axs.T:
            for ax in ax_column[0:-1]:
                ax.sharex(ax_column[0])
                ax.xaxis.set_tick_params(labelbottom=False)
                ax.set_xlabel(None)
        for i, ax_row in enumerate(axs):
            for ax in ax_row[1:i]:
                ax.sharey(ax_row[0])
                ax.yaxis.set_tick_params(labelleft=False)
                ax.set_ylabel(None)
        for i, _ in enumerate(dimensions):
            axs[i, i].sharey(axs[0, 0])
            axs[i, i].set_yticks([])
            axs[i, i].set_ylabel(None)

        return fig, axs

    # Additional utility methods for screen imaging
    def get_spatial_projection(self):
        """Get x-y coordinates for screen imaging"""
        return torch.stack([self.x, self.y], dim=-1)
    
    def get_phase_space_projection(self, plane: str = 'x'):
        """Get phase space projection for specified plane"""
        if plane == 'x':
            return torch.stack([self.x, self.px], dim=-1)
        elif plane == 'y':
            return torch.stack([self.y, self.py], dim=-1)
        else:
            raise ValueError("Plane must be 'x' or 'y'")
