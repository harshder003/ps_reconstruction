import torch
from scipy.stats import gaussian_kde
from typing import Optional
from cheetah.particles import ParticleBeam
from .beams import TransverseBeam4D
from cheetah import Screen

def generate_screen_image_with_kde(beam, pixel_size=1e-3, image_size=100, kernel_bandwidth=None, device=None, batch_size=5):
    """
    Generate a screen image from a particle beam using Kernel Density Estimation with PyTorch.
    Memory-efficient implementation that processes data in small batches.
    
    Args:
        beam (ParticleBeam or TransverseBeam4D): The beam to image
        pixel_size (float): Size of each pixel in meters
        image_size (int): Number of pixels per dimension in the output image
        kernel_bandwidth (float, optional): Bandwidth for the KDE. If None, it's estimated
        device (torch.device, optional): Device to perform computations on
        batch_size (int): Number of rows to process at once
        
    Returns:
        torch.Tensor: 2D image array
    """
    # Use beam's device if none specified
    if device is None:
        device = beam.x.device
        
    # Clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Extract x and y coordinates directly
    positions = torch.stack((beam.x, beam.y), dim=1)
    
    # Ensure positions are on the correct device
    if positions.device != device:
        positions = positions.to(device)
    
    # Filter out any NaN or infinite values in a single operation
    valid_indices = torch.isfinite(positions).all(dim=1)
    valid_positions = positions[valid_indices]
    
    if valid_positions.shape[0] < 2:
        return torch.zeros((image_size, image_size), device=device)
    
    # If bandwidth is None, estimate it using Scott's rule
    if kernel_bandwidth is None:
        # Scott's rule: n**(-1/(d+4)) where n is sample size and d is dimensions
        n = valid_positions.shape[0]
        d = valid_positions.shape[1]
        bandwidth = n**(-1.0/(d+4)) * valid_positions.std(dim=0).mean()
    else:
        bandwidth = kernel_bandwidth
        
    # Ensure bandwidth is a tensor on the correct device
    if not isinstance(bandwidth, torch.Tensor):
        bandwidth = torch.tensor(bandwidth, device=device)
    elif bandwidth.device != device:
        bandwidth = bandwidth.to(device)
    
    # Create bin centers for evaluation
    half_size = (image_size * pixel_size) / 2
    x_bins = torch.linspace(-half_size, half_size, image_size, device=device)
    y_bins = torch.linspace(-half_size, half_size, image_size, device=device)
    
    # Extract the valid x and y values
    x_values = valid_positions[:, 0]
    y_values = valid_positions[:, 1]
    
    # Check if particle_charges and survival_probabilities exist and have proper dimensions
    weights = None
    try:
        # Check if these attributes exist and are properly dimensioned tensors
        if (hasattr(beam, 'particle_charges') and 
            hasattr(beam, 'survival_probabilities') and
            beam.particle_charges.dim() > 0 and 
            beam.survival_probabilities.dim() > 0):
            
            weights = beam.particle_charges[valid_indices].abs() * beam.survival_probabilities[valid_indices]
    except (AttributeError, IndexError, TypeError):
        # If any error occurs, just use uniform weights
        weights = None
    
    # Process in smaller batches for both grid points and positions
    density = torch.zeros((image_size, image_size), device=device)
    
    # Process the grid points row by row to minimize memory usage
    for i in range(0, image_size, batch_size):
        end_i = min(i + batch_size, image_size)
        
        # Process this batch of rows
        y_batch = y_bins[i:end_i]
        
        # Initialize density for these rows
        batch_density = torch.zeros((end_i - i, image_size), device=device)
        
        # Process particles in batches
        particle_batch_size = min(1000, valid_positions.shape[0])
        for k in range(0, valid_positions.shape[0], particle_batch_size):
            end_k = min(k + particle_batch_size, valid_positions.shape[0])
            
            # Get batch of particles
            x_batch = x_values[k:end_k]
            y_batch_particles = y_values[k:end_k]
            
            # Get batch weights if available
            batch_weights = None if weights is None else weights[k:end_k]
            
            # For each row in the current batch
            for row_idx, y_val in enumerate(y_batch):
                # Calculate y-dimension kernel values
                y_diff = y_batch_particles.unsqueeze(1) - y_val
                y_kernel = torch.exp(-0.5 * (y_diff / bandwidth)**2) / (bandwidth * (2 * torch.pi)**0.5)
                
                # For each particle, calculate its contribution to all x positions
                for x_idx, x_val in enumerate(x_bins):
                    # Calculate x-dimension kernel values
                    x_diff = x_batch - x_val
                    x_kernel = torch.exp(-0.5 * (x_diff / bandwidth)**2) / (bandwidth * (2 * torch.pi)**0.5)
                    
                    # Combine kernels and apply weights
                    if batch_weights is not None:
                        kernel_values = x_kernel * y_kernel.squeeze(1) * batch_weights
                    else:
                        kernel_values = x_kernel * y_kernel.squeeze(1)
                    
                    # Add to density
                    batch_density[row_idx, x_idx] += kernel_values.sum()
        
        # Normalize the batch
        normalization_factor = valid_positions.shape[0] * (2 * torch.pi * bandwidth**2)**0.5
        batch_density /= normalization_factor
        
        # Update the main density tensor
        density[i:end_i] = batch_density
        
        # Free memory
        del batch_density
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return density

def generate_image_with_cheetah_kde(
    beam: TransverseBeam4D | ParticleBeam,
    pixel_size: float = 1e-3,
    image_size: int = 100,
    kde_bandwidth: Optional[float] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate screen image using Cheetah's optimized Screen element.
    
    This is orders of magnitude faster than manual KDE implementation.
    """
    if device is None:
        device = beam.particles.device
    
    # Create Cheetah Screen with KDE method
    screen_kwargs = {
        'resolution': (image_size, image_size),
        'pixel_size': torch.tensor([pixel_size, pixel_size], device=device),
        'method': 'kde',
        'is_active': True,
        'device': device
    }
    
    if kde_bandwidth is not None:
        screen_kwargs['kde_bandwidth'] = torch.tensor(kde_bandwidth, device=device)
    
    screen = Screen(**screen_kwargs)
    
    # Track beam through screen (this generates the image)
    _ = screen.track(beam)
    
    # Return the generated image
    return screen.reading

def generate_screen_image_with_histogram(beam, bins=100, range_factor=2.5):
    """
    Generate a screen image from a particle beam using histograms.
    
    Args:
        beam (ParticleBeam or TransverseBeam4D): The beam to image
        bins (int): Number of bins in each dimension
        range_factor (float): Factor to multiply beam standard deviation by to determine range
        
    Returns:
        torch.Tensor: 2D histogram array of the beam distribution
    """
    # Extract positions and filter invalids
    positions = torch.stack((beam.x, beam.y), dim=1)
    valid_indices = torch.isfinite(positions).all(dim=1)
    valid_x = positions[valid_indices, 0]
    valid_y = positions[valid_indices, 1]
    
    if len(valid_x) < 2:
        return torch.zeros((bins, bins))
    
    # Calculate beam statistics
    x_mean, y_mean = valid_x.mean(), valid_y.mean()
    x_std, y_std = valid_x.std(), valid_y.std()
    
    # Set histogram range based on beam statistics
    x_range = (x_mean - range_factor * x_std, x_mean + range_factor * x_std)
    y_range = (y_mean - range_factor * y_std, y_mean + range_factor * y_std)
    
    # Create the 2D histogram
    hist_values, x_edges, y_edges = torch.histogram2d(
        valid_x, valid_y, 
        bins=bins,
        range=[x_range, y_range]
    )
    
    return hist_values