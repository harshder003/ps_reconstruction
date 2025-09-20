import torch
import torch.nn.functional as F
import gc
import numpy as np # Import numpy for type checking in Dataset

class ScreenImageGenerator:
    """
    GPU-accelerated class for generating screen images using PyTorch and FFT optimizations.
    """
    
    def __init__(self, r11, r12, r33, r34, quad_k, x_pixels, y_pixels, x_pixel_size, y_pixel_size, device=None, particle_batch_size=2000, config_batch_size=None):
        """
        Initialize the GPU-accelerated screen image generator.
        
        Parameters:
        -----------
        r11, r12, r33, r34 : array-like
            Transfer matrix elements
        quad_k : array-like
            Quadrupole strengths
        x_pixels, y_pixels : int
            Number of pixels in x and y dimensions
        x_pixel_size, y_pixel_size : float
            Pixel size in meters for x and y dimensions
        device : str or torch.device
            Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Convert to tensors and move to device. These are fixed parameters from data, so no grad needed.
        self.r11 = torch.tensor(r11, dtype=torch.float32, device=self.device)
        self.r12 = torch.tensor(r12, dtype=torch.float32, device=self.device)
        self.r33 = torch.tensor(r33, dtype=torch.float32, device=self.device)
        self.r34 = torch.tensor(r34, dtype=torch.float32, device=self.device)
        self.quad_k = torch.tensor(quad_k, dtype=torch.float32, device=self.device) # Store as tensor for consistent device
        
        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        
        # Calculate screen dimensions
        self.x_extent = x_pixels * x_pixel_size
        self.y_extent = y_pixels * y_pixel_size
        
        # Pre-compute pixel coordinate arrays on GPU
        # These define the centers of the pixels
        self.x_screen_coords = torch.linspace(-self.x_extent/2, self.x_extent/2, x_pixels, device=self.device)
        self.y_screen_coords = torch.linspace(-self.y_extent/2, self.y_extent/2, y_pixels, device=self.device)

        self.particle_batch_size = particle_batch_size
        self.config_batch_size = config_batch_size or 1
        
    def _differentiable_splat(self, x_screen: torch.Tensor, y_screen: torch.Tensor, splat_sigma: float):
        """
        Memory-efficient differentiable splatting using particle batching.
        """
        if x_screen.numel() == 0:
            return torch.zeros(self.y_pixels, self.x_pixels, device=self.device, requires_grad=True)

        # Initialize empty image
        image = torch.zeros(self.y_pixels, self.x_pixels, device=self.device, requires_grad=True)

        num_particles = x_screen.shape[0]

        # Process particles in batches
        for batch_start in range(0, num_particles, self.particle_batch_size):
            batch_end = min(batch_start + self.particle_batch_size, num_particles)

            x_batch = x_screen[batch_start:batch_end]
            y_batch = y_screen[batch_start:batch_end]

            if x_batch.numel() == 0:
                continue

            # Compute for this batch
            x_screen_p = x_batch.unsqueeze(1).unsqueeze(2)
            y_screen_p = y_batch.unsqueeze(1).unsqueeze(2)

            x_grid_centers = self.x_screen_coords.view(1, 1, -1)
            y_grid_centers = self.y_screen_coords.view(1, -1, 1)

            result_dist_sq = (x_screen_p - x_grid_centers)**2 + (y_screen_p - y_grid_centers)**2
            kernel_values = torch.exp(-result_dist_sq / (2 * (splat_sigma + 1e-8)**2))

            # Accumulate into the main image
            image = image + kernel_values.sum(dim=0)

            # Clear intermediate tensors
            del x_screen_p, y_screen_p, result_dist_sq, kernel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Normalization
        total_charge = image.sum()
        if total_charge > 1e-8:
            image = image / total_charge * x_screen.shape[0]

        return image
    
    def _create_gaussian_kernel(self, sigma_in_pixels: float, kernel_size: int = None):
        """
        Create Gaussian kernel for FFT-based convolution.
        
        Parameters:
        -----------
        sigma_in_pixels : float
            Standard deviation of the Gaussian kernel, expressed in *pixel units*.
        kernel_size : int, optional
            Explicit size for the kernel. If None, it's derived from sigma.
        """
        if kernel_size is None:
            # The sigma is already in pixel units, so just use it directly.
            # No division by self.x_pixel_size needed here.
            kernel_size = int(6 * sigma_in_pixels + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, kernel_size) # Ensure a minimum reasonable kernel size
        
        # Create 1D Gaussian
        x = torch.arange(kernel_size, device=self.device, dtype=torch.float32)
        x = x - kernel_size // 2
        
        # Use sigma_in_pixels directly in the Gaussian formula.
        # Add a small epsilon to prevent division by zero if sigma_in_pixels is exactly 0.
        gaussian_1d = torch.exp(-0.5 * (x / (sigma_in_pixels + 1e-8)) ** 2)
        gaussian_1d = gaussian_1d / gaussian_1d.sum() # Normalize so it sums to 1

        # Create 2D Gaussian kernel
        kernel = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        
        return kernel
    
    def _fft_convolve2d(self, image, kernel):
        """FFT-based 2D convolution for Gaussian blur."""
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        
        output_size_h = img_h + ker_h - 1
        output_size_w = img_w + ker_w - 1
        
        fft_h = int(2**torch.ceil(torch.log2(torch.tensor(output_size_h, dtype=torch.float32))))
        fft_w = int(2**torch.ceil(torch.log2(torch.tensor(output_size_w, dtype=torch.float32))))

        image_padded = F.pad(image.unsqueeze(0).unsqueeze(0), 
                           (0, fft_w - img_w, 0, fft_h - img_h), mode='constant', value=0)
        kernel_padded = F.pad(kernel.unsqueeze(0).unsqueeze(0),
                            (0, fft_w - ker_w, 0, fft_h - ker_h),
                            mode='constant', value=0)
        
        image_fft = torch.fft.fft2(image_padded)
        kernel_fft = torch.fft.fft2(kernel_padded)
        result_fft = image_fft * kernel_fft
        result = torch.fft.ifft2(result_fft).real
        
        pad_h = ker_h // 2
        pad_w = ker_w // 2
        result = result[0, 0, pad_h : pad_h + img_h, pad_w : pad_w + img_w]
        
        return result
    
    def print_memory(step_name):
        """Helper function to print current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"[MEMORY] {step_name}: Allocated={allocated:.3f}GB, Reserved={reserved:.3f}GB")
        else:
            print(f"[MEMORY] {step_name}: CPU mode")

    def generate_screen_image(self, beam, gaussian_blur_sigma=0.5, 
                            use_kde=True, batch_size=None):
        """
        Generate screen images from beam using GPU acceleration with memory optimization.
        Processes particles in batches and uses gradient checkpointing to avoid memory issues.
        """
        import gc
        from torch.utils.checkpoint import checkpoint
    
        # Convert beam coordinates to tensors
        if hasattr(beam.x, 'detach'):
            x_coords = beam.x.to(self.device)
            px_coords = beam.px.to(self.device)
            y_coords = beam.y.to(self.device)
            py_coords = beam.py.to(self.device)
        else:
            x_coords = torch.tensor(beam.x, dtype=torch.float32, device=self.device)
            px_coords = torch.tensor(beam.px, dtype=torch.float32, device=self.device)
            y_coords = torch.tensor(beam.y, dtype=torch.float32, device=self.device)
            py_coords = torch.tensor(beam.py, dtype=torch.float32, device=self.device)
            print("Warning: Beam coordinates were not torch.Tensor. Ensure gradient tracking is maintained.")
    
        splat_sigma_in_meters = 0.5 * self.x_pixel_size
    
        gaussian_kernel = None
        if gaussian_blur_sigma > 0:
            gaussian_kernel = self._create_gaussian_kernel(gaussian_blur_sigma, kernel_size=4)
    
        images_output_list = []
        n_configs = len(self.r11)
        n_particles = x_coords.shape[0]
        
        # CRITICAL: Use smaller particle batches to prevent memory explosion
        particle_batch_size = min(2000, n_particles)  # Reduce if still getting OOM
        
        def process_single_config_with_particle_batching(config_idx):
            """Process a single configuration with particle batching."""
            # Initialize accumulator for this configuration
            final_image = torch.zeros(self.y_pixels, self.x_pixels, device=self.device, requires_grad=True)
            
            total_valid_particles = 0
            
            # Process particles in batches
            for p_start in range(0, n_particles, particle_batch_size):
                p_end = min(p_start + particle_batch_size, n_particles)
                
                # Get particle batch
                x_batch = x_coords[p_start:p_end]
                px_batch = px_coords[p_start:p_end]
                y_batch = y_coords[p_start:p_end]
                py_batch = py_coords[p_start:p_end]
                
                # Compute screen coordinates for this particle batch
                x_screen = x_batch * self.r11[config_idx] + px_batch * self.r12[config_idx]
                y_screen = y_batch * self.r33[config_idx] + py_batch * self.r34[config_idx]
                
                # Apply valid mask
                valid_mask = (torch.abs(x_screen) < self.x_extent/2) & (torch.abs(y_screen) < self.y_extent/2)
                x_screen_valid = x_screen[valid_mask]
                y_screen_valid = y_screen[valid_mask]
                
                total_valid_particles += x_screen_valid.shape[0]
                
                # Clear intermediate variables immediately
                del x_batch, px_batch, y_batch, py_batch, x_screen, y_screen, valid_mask
                
                # Only process if we have valid particles
                if x_screen_valid.shape[0] > 0:
                    # Use gradient checkpointing to reduce memory while preserving gradients
                    batch_image = checkpoint(
                        self._differentiable_splat,
                        x_screen_valid, 
                        y_screen_valid, 
                        splat_sigma_in_meters,
                        use_reentrant=False  # Use new checkpointing API
                    )
                    
                    # Accumulate into final image
                    final_image = final_image + batch_image
                    del batch_image
                
                # Clear particle batch variables
                del x_screen_valid, y_screen_valid
                
                # Aggressive cleanup after each particle batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Normalize by total number of particles (not just valid ones)
            if total_valid_particles > 0:
                final_image = final_image * (n_particles / total_valid_particles)
            
            return final_image
    
        # Process each configuration individually to avoid large batch tensors
        for config_idx in range(n_configs):
            # Process with particle batching and gradient checkpointing
            image = process_single_config_with_particle_batching(config_idx)
            
            # Apply Gaussian blur if needed
            if gaussian_kernel is not None:
                image = self._fft_convolve2d(image, gaussian_kernel)
    
            # Break computation graph and add to list
            image = image.detach().clone()
            images_output_list.append(image)
    
            # Aggressive memory cleanup after each configuration
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
        return images_output_list
    
    def generate_screen_image_with_params(self, beam, r11, r12, r33, r34, quad_k,
                                    gaussian_blur_sigma=0.5, use_kde=True, batch_size=None):
        """
        Memory-optimized version using particle batching and gradient checkpointing.
        """
        import gc
        from torch.utils.checkpoint import checkpoint
        
        # Convert beam coordinates to tensors
        if hasattr(beam.x, 'detach'):
            x_coords = beam.x.to(self.device)
            px_coords = beam.px.to(self.device)
            y_coords = beam.y.to(self.device)
            py_coords = beam.py.to(self.device)
        else:
            x_coords = torch.tensor(beam.x, dtype=torch.float32, device=self.device)
            px_coords = torch.tensor(beam.px, dtype=torch.float32, device=self.device)
            y_coords = torch.tensor(beam.y, dtype=torch.float32, device=self.device)
            py_coords = torch.tensor(beam.py, dtype=torch.float32, device=self.device)
            print("Warning: Beam coordinates were not torch.Tensor. Ensure gradient tracking is maintained.")
    
        # Ensure parameters are on correct device
        r11_tensor = r11.to(self.device)
        r12_tensor = r12.to(self.device)
        r33_tensor = r33.to(self.device)
        r34_tensor = r34.to(self.device)
        
        splat_sigma_in_meters = 0.5 * self.x_pixel_size
    
        gaussian_kernel = None
        if gaussian_blur_sigma > 0:
            gaussian_kernel = self._create_gaussian_kernel(gaussian_blur_sigma, kernel_size=4)
        
        images_output_list = []
        n_configs = len(r11_tensor)
        n_particles = x_coords.shape[0]
        
        # CRITICAL: Use smaller particle batches to prevent memory explosion
        particle_batch_size = min(2000, n_particles)  # Reduce if still getting OOM
        
        def process_single_config_with_particle_batching(config_idx):
            """Process a single configuration with particle batching."""
            # Initialize accumulator for this configuration
            final_image = torch.zeros(self.y_pixels, self.x_pixels, device=self.device, requires_grad=True)
            
            total_valid_particles = 0
            
            # Process particles in batches
            for p_start in range(0, n_particles, particle_batch_size):
                p_end = min(p_start + particle_batch_size, n_particles)
                
                # Get particle batch
                x_batch = x_coords[p_start:p_end]
                px_batch = px_coords[p_start:p_end]
                y_batch = y_coords[p_start:p_end]
                py_batch = py_coords[p_start:p_end]
                
                # Compute screen coordinates for this particle batch
                x_screen = x_batch * r11_tensor[config_idx] + px_batch * r12_tensor[config_idx]
                y_screen = y_batch * r33_tensor[config_idx] + py_batch * r34_tensor[config_idx]
                
                # Apply valid mask
                valid_mask = (torch.abs(x_screen) < self.x_extent/2) & (torch.abs(y_screen) < self.y_extent/2)
                x_screen_valid = x_screen[valid_mask]
                y_screen_valid = y_screen[valid_mask]
                
                total_valid_particles += x_screen_valid.shape[0]
                
                # Clear intermediate variables immediately
                del x_batch, px_batch, y_batch, py_batch, x_screen, y_screen, valid_mask
                
                # Only process if we have valid particles
                if x_screen_valid.shape[0] > 0:
                    # Use gradient checkpointing to reduce memory while preserving gradients
                    batch_image = checkpoint(
                        self._differentiable_splat,
                        x_screen_valid, 
                        y_screen_valid, 
                        splat_sigma_in_meters,
                        use_reentrant=False  # Use new checkpointing API
                    )
                    
                    # Accumulate into final image
                    final_image = final_image + batch_image
                    del batch_image
                
                # Clear particle batch variables
                del x_screen_valid, y_screen_valid
                
                # Aggressive cleanup after each particle batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Normalize by total number of particles (not just valid ones)
            if total_valid_particles > 0:
                final_image = final_image * (n_particles / total_valid_particles)
            
            return final_image
        
        # Process each configuration
        for config_idx in range(n_configs):
            # Process with particle batching and gradient checkpointing
            image = process_single_config_with_particle_batching(config_idx)
            
            # Apply Gaussian blur if needed
            if gaussian_kernel is not None:
                image = self._fft_convolve2d(image, gaussian_kernel)
            
            # Add to results
            images_output_list.append(image)
            
            # Cleanup after each configuration
            del image
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return images_output_list

    def generate_single_image(self, beam, config_index=0, gaussian_blur_sigma=0.5):
        """Generate image for a single configuration with GPU acceleration using differentiable splatting."""
        if hasattr(beam.x, 'detach'):
            x_coords = beam.x.to(self.device)
            px_coords = beam.px.to(self.device)
            y_coords = beam.y.to(self.device)
            py_coords = beam.py.to(self.device)
        else:
            x_coords = torch.tensor(beam.x, dtype=torch.float32, device=self.device)
            px_coords = torch.tensor(beam.px, dtype=torch.float32, device=self.device)
            y_coords = torch.tensor(beam.y, dtype=torch.float32, device=self.device)
            py_coords = torch.py.to(self.device)
            print("Warning: Beam coordinates were not torch.Tensor. Ensure gradient tracking is maintained.")
        
        r11, r12, r33, r34 = (self.r11[config_index], self.r12[config_index], 
                              self.r33[config_index], self.r34[config_index])
        
        x_screen = x_coords * r11 + px_coords * r12
        y_screen = y_coords * r33 + py_coords * r34
        
        valid_mask = (torch.abs(x_screen) < self.x_extent/2) & (torch.abs(y_screen) < self.y_extent/2)
        x_screen = x_screen[valid_mask]
        y_screen = y_screen[valid_mask]
        
        splat_sigma_in_meters = 0.5 * self.x_pixel_size

        image = self._differentiable_splat(x_screen, y_screen, splat_sigma_in_meters)
        
        if gaussian_blur_sigma > 0:
            gaussian_kernel = self._create_gaussian_kernel(gaussian_blur_sigma)
            image = self._fft_convolve2d(image, gaussian_kernel)
        
        return image # Return single tensor
    
    def clear_cache(self):
        """Clear GPU cache and perform garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache()