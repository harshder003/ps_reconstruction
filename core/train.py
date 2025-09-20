import os
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from .beam_generator import NNParticleBeamGenerator, ParticleTransformer, NNTransform
from .simulation import ScreenImageGenerator
from .losses import ImageSimilarityLoss, normalize_images

class LitBeamImageGenerator(L.LightningModule):
    def __init__(
        self, 
        beam_generator: NNParticleBeamGenerator,
        screen_generator: ScreenImageGenerator,
        lr: float = 1e-3,
        loss_mode: str = "weighted",
        foreground_weight: float = 20.0,
        background_weight: float = 0.5,
        gaussian_blur_sigma: float = 0.5
    ):
        """
        Lightning module for training beam generator to match target images.
        
        Parameters:
        -----------
        beam_generator : NNParticleBeamGenerator
            The neural network beam generator to train
        screen_generator : ScreenImageGenerator  
            GPU-accelerated screen image generator
        lr : float
            Learning rate for optimizer
        loss_mode : str
            Loss function mode ("weighted", "mse", "l1")
        foreground_weight : float
            Weight for foreground pixels in weighted loss
        background_weight : float
            Weight for background pixels in weighted loss
        gaussian_blur_sigma : float
            Gaussian blur sigma for generated images
        """
        super().__init__()
        self.beam_generator = beam_generator
        self.screen_generator = screen_generator
        self.lr = lr
        self.gaussian_blur_sigma = gaussian_blur_sigma
        
        # Loss function
        self.image_similarity_loss = ImageSimilarityLoss(
            mode=loss_mode, 
            foreground_weight=foreground_weight, 
            background_weight=background_weight
        )
        
        # For logging
        self.epoch_losses = []
        
    def _prepare_target_images(self, target_images):
        """Convert target images to tensor format"""
        images_tensor = []
        for img_data in target_images:
            if isinstance(img_data, list) and len(img_data) >= 1:
                # Extract just the image array
                img = img_data[0]
            else:
                img = img_data
            
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32)
            images_tensor.append(img)
        
        return torch.stack(images_tensor)
    
    def forward(self, r11=None, r12=None, r33=None, r34=None, quad_k=None):
        """
        Generate beam and create screen images with provided R-matrix parameters.
        
        Parameters:
        -----------
        r11, r12, r33, r34 : torch.Tensor, optional
            Transfer matrix elements for each batch item
        quad_k : torch.Tensor, optional
            Quadrupole strengths for each batch item
            
        Returns:
        --------
        torch.Tensor
            Generated screen images
        """
        # Generate beam using the neural network
        beam = self.beam_generator()
        
        if r11 is not None:
            # Use provided R-matrix parameters
            # Handle batch dimension - process each item in batch
            batch_size = r11.shape[0]
            all_images = []
            
            
            for i in range(batch_size):
                # Extract R-matrix parameters for this batch item
                r11_i = r11[i]
                r12_i = r12[i]
                r33_i = r33[i]
                r34_i = r34[i]
                quad_k_i = quad_k[i]
                
                # Generate screen images using provided parameters
                image_tensors = self.screen_generator.generate_screen_image_with_params(
                    beam, 
                    r11=r11_i,
                    r12=r12_i,
                    r33=r33_i,
                    r34=r34_i,
                    quad_k=quad_k_i,
                    gaussian_blur_sigma=self.gaussian_blur_sigma
                )
                
                # # Stack images for this batch item
                batch_images = torch.stack(image_tensors)
                all_images.append(batch_images)
            
            # Stack all batch items
            return torch.stack(all_images)
        else:
            # Use default R-matrix parameters from screen generator
            generated_images = self.screen_generator.generate_screen_image(
                beam, 
                gaussian_blur_sigma=self.gaussian_blur_sigma
            )
            
            # Extract just the image arrays and convert to tensor
            image_tensors = []
            for img_data in generated_images:
                img = img_data[0]  # Extract image array
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(img, dtype=torch.float32, device=self.device)
                else:
                    img = img.to(self.device)
                image_tensors.append(img)
            
            return torch.stack(image_tensors).unsqueeze(0)  # Add batch dimension
    
    def training_step(self, batch, batch_idx):
        """Training step with batch data containing R-matrix parameters"""
        # Extract data from batch
        target_images = batch['images'].to(self.device)
        r11 = batch['r11'].to(self.device)
        r12 = batch['r12'].to(self.device)
        r33 = batch['r33'].to(self.device)
        r34 = batch['r34'].to(self.device)
        quad_k = batch['quad_k'].to(self.device)
        
        # Generate predicted images using provided R-matrix parameters
        pred_images = self.forward(r11=r11, r12=r12, r33=r33, r34=r34, quad_k=quad_k)
        
        # Handle batch dimensions for loss calculation
        batch_size = target_images.shape[0]
        total_loss = 0.0
        
        for i in range(batch_size):
            # Get images for this batch item
            pred_batch = pred_images[i]  # [n_configs, H, W]
            target_batch = target_images[i]  # Should be [n_configs, H, W] or [H, W]
            
            # Handle case where target is single image
            if target_batch.dim() == 2:
                target_batch = target_batch.unsqueeze(0)  # Add config dimension
            
            # Ensure same number of configurations
            if pred_batch.shape[0] != target_batch.shape[0]:
                # If target has only one image, repeat it
                if target_batch.shape[0] == 1:
                    target_batch = target_batch.repeat(pred_batch.shape[0], 1, 1)
                else:
                    raise ValueError(f"Mismatch in number of configurations: pred={pred_batch.shape[0]}, target={target_batch.shape[0]}")
            
            # Normalize images
            pred_normalized = normalize_images(pred_batch)
            target_normalized = normalize_images(target_batch)
            
            # Compute loss for this batch item
            loss = self.image_similarity_loss(pred_normalized, target_normalized)
            total_loss += loss
        
        # Average loss over batch
        avg_loss = total_loss / batch_size
        
        # Log loss
        self.log("train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return avg_loss
    
    def on_train_epoch_end(self):
        """Log loss every 10 epochs"""
        current_epoch = self.current_epoch
        if current_epoch % 10 == 0:
            avg_loss = self.trainer.callback_metrics.get("train_loss_epoch", 0.0)
            print(f"Epoch {current_epoch}: Average Loss = {avg_loss:.6f}")
            self.epoch_losses.append((current_epoch, float(avg_loss)))

            # Set model to evaluation mode
            self.eval() 
            with torch.no_grad():
                try:
                    sample_batch = next(iter(self.trainer.train_dataloader))
                except StopIteration:
                    # If the dataloader is exhausted (e.g., small dataset), re-initialize iterator
                    self.trainer.train_dataloader.dataset.reset_iterator() # Add this method to your BeamImageDataset
                    sample_batch = next(iter(self.trainer.train_dataloader))
                
                # Move sample batch data to the device
                sample_target_images = sample_batch['images'].to(self.device)
                sample_r11 = sample_batch['r11'].to(self.device)
                sample_r12 = sample_batch['r12'].to(self.device)
                sample_r33 = sample_batch['r33'].to(self.device)
                sample_r34 = sample_batch['r34'].to(self.device)
                sample_quad_k = sample_batch['quad_k'].to(self.device)

                # Generate predicted images for this sample batch
                # Output shape: [BATCH_SIZE, N_CONFIGS, H, W]
                pred_images_sample = self.forward(
                    r11=sample_r11,
                    r12=sample_r12,
                    r33=sample_r33,
                    r34=sample_r34,
                    quad_k=sample_quad_k
                )
                # Move to CPU and convert to NumPy, then normalize
                # We'll take the first item from the batch for visualization
                vis_pred_images = normalize_images(pred_images_sample[:,0]).cpu().numpy()
                vis_target_images = normalize_images(sample_target_images).cpu().numpy()

                # Determine how many images to display (e.g., up to 5)
                num_to_display = min(vis_pred_images.shape[1], 10)

                # Setup directory for saving images
                log_path = self.logger.log_dir
                epoch_img_dir = os.path.join(log_path, "epoch_images", f"epoch_{current_epoch:04d}")
                os.makedirs(epoch_img_dir, exist_ok=True)

                for i in range(num_to_display):
                    if (i+1)%3!=0:
                        continue
                    pred_img = vis_pred_images[i]
                    target_img = vis_target_images[i]
                    
                    # Plotting predicted vs. target image
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    im0 = axes[0].imshow(pred_img, cmap='hot', origin='lower')
                    axes[0].set_title(f"Epoch {current_epoch} - Predicted Image {i+1}")
                    axes[0].axis('off')
                    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

                    im1 = axes[1].imshow(target_img, cmap='hot', origin='lower')
                    axes[1].set_title(f"Epoch {current_epoch} - Target Image {i+1}")
                    axes[1].axis('off')
                    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(epoch_img_dir, f"image_comparison_config_{i+1}.png"))
                    plt.close(fig) # Close the figure to free memory
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",
            },
        }

def create_dummy_dataloader(batch_size=1, num_batches=100):
    """Create a dummy dataloader since we don't need real batched data"""
    # Create dummy data - we don't actually use this in training
    dummy_data = torch.zeros((num_batches * batch_size, 1))
    dummy_targets = torch.zeros((num_batches * batch_size, 1))
    
    dataset = TensorDataset(dummy_data, dummy_targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_beam_generator_with_dataloader(
    beam_generator: NNParticleBeamGenerator,
    screen_generator: ScreenImageGenerator,
    train_dataloader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    loss_mode: str = "weighted",
    foreground_weight: float = 20.0,
    background_weight: float = 0.5,
    gaussian_blur_sigma: float = 0.5,
    log_dir: str = "logs",
    experiment_name: str = "beam_training",
    checkpoint_every: int = 50,
    device: str = None,
    **trainer_kwargs
):
    """
    Train the beam generator using a proper DataLoader.
    
    Parameters:
    -----------
    beam_generator : NNParticleBeamGenerator
        The beam generator model to train
    screen_generator : ScreenImageGenerator
        GPU-accelerated screen image generator
    train_dataloader : DataLoader
        DataLoader containing target images and R-matrix parameters
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    loss_mode : str
        Loss function mode ("weighted", "mse", "l1")
    foreground_weight : float
        Weight for foreground pixels in weighted loss
    background_weight : float
        Weight for background pixels in weighted loss
    gaussian_blur_sigma : float
        Gaussian blur sigma for generated images
    log_dir : str
        Directory for logging
    experiment_name : str
        Name for the experiment
    checkpoint_every : int
        Save checkpoint every N epochs
    device : str
        Device to use ('cuda', 'cpu', or None for auto)
    **trainer_kwargs
        Additional arguments for Lightning Trainer
        
    Returns:
    --------
    trained_model : LitBeamImageGenerator
        The trained Lightning module
    """
    
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on device: {device}")
    print(f"Training for {n_epochs} epochs with lr={lr}")
    
    # Create Lightning module
    lit_model = LitBeamImageGenerator(
        beam_generator=beam_generator,
        screen_generator=screen_generator,
        lr=lr,
        loss_mode=loss_mode,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        gaussian_blur_sigma=gaussian_blur_sigma
    )
    
    # Set up logger
    logger = CSVLogger(log_dir, name=experiment_name)
    
    # Set up checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="beam_generator_{epoch:04d}",
        every_n_epochs=checkpoint_every,
        save_top_k=-1,  # Save all checkpoints
        monitor="train_loss_epoch",
        mode="min"
    )
    
    # Set up trainer
    trainer = L.Trainer(
        max_epochs=n_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=1,
        **trainer_kwargs
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(lit_model, train_dataloader)
    
    print("Training completed!")
    print(f"Logs saved to: {logger.log_dir}")
    
    # Print final loss summary
    if lit_model.epoch_losses:
        print("\nLoss summary (every 10 epochs):")
        for epoch, loss in lit_model.epoch_losses:
            print(f"  Epoch {epoch}: {loss:.6f}")
    
    return lit_model

def continue_training(
    trained_model: LitBeamImageGenerator,
    train_dataloader: DataLoader,
    additional_epochs: int = 20,
    checkpoint_every: int = 50,
    **trainer_kwargs
):
    """
    Continue training a previously trained model for additional epochs.
    
    Parameters:
    -----------
    trained_model : LitBeamImageGenerator
        The previously trained Lightning module
    train_dataloader : DataLoader
        DataLoader containing target images and R-matrix parameters
    additional_epochs : int
        Number of additional epochs to train
    checkpoint_every : int
        Save checkpoint every N epochs
    **trainer_kwargs
        Additional arguments for Lightning Trainer
        
    Returns:
    --------
    trained_model : LitBeamImageGenerator
        The model after additional training
    """
    
    # Get the current epoch from the trained model
    current_epoch = trained_model.current_epoch
    new_max_epochs = current_epoch + additional_epochs
    
    print(f"Continuing training from epoch {current_epoch}")
    print(f"Training for {additional_epochs} additional epochs (until epoch {new_max_epochs})")
    
    # Get the existing logger and checkpoint callback from the trainer
    existing_logger = trained_model.logger
    existing_log_dir = existing_logger.log_dir
    
    print(f"Continuing to log in: {existing_log_dir}")
    
    # Set up new checkpointing (continuing in same directory)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(existing_log_dir, "checkpoints"),
        filename="beam_generator_{epoch:04d}",
        every_n_epochs=checkpoint_every,
        save_top_k=-1,  # Save all checkpoints
        monitor="train_loss_epoch",
        mode="min"
    )
    
    # Determine device from model
    device = next(trained_model.parameters()).device
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    
    # Set up new trainer with same logger
    trainer = L.Trainer(
        max_epochs=new_max_epochs,
        logger=existing_logger,  # Use the same logger to continue in same version
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=1,
        **trainer_kwargs
    )
    
    # Continue training
    print("Resuming training...")
    trainer.fit(trained_model, train_dataloader)
    
    print("Additional training completed!")
    print(f"Total epochs trained: {trained_model.current_epoch}")
    
    # Print updated loss summary
    if trained_model.epoch_losses:
        print("\\nUpdated loss summary (every 10 epochs):")
        for epoch, loss in trained_model.epoch_losses:
            print(f"  Epoch {epoch}: {loss:.6f}")
    
    return trained_model

