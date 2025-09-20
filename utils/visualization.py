import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from ..core.losses import normalize_images

def align_images_cross_correlation_cuda(img1, img2):
    """Align img2 to img1 using cross-correlation on CUDA with FFT optimization."""
    # Move to CUDA
    img1_cuda = torch.tensor(img1, device='cuda', dtype=torch.float32)
    img2_cuda = torch.tensor(img2, device='cuda', dtype=torch.float32)

    # Compute cross-correlation using FFT for O(n log n) complexity instead of O(n^2)
    pad_size = [s*2-1 for s in img1_cuda.shape]
    f1 = torch.fft.fftn(img1_cuda, s=pad_size)
    f2 = torch.fft.fftn(img2_cuda, s=pad_size)
    correlation = torch.fft.ifftn(f1 * torch.conj(f2)).real

    # Find the peak of the cross-correlation
    max_idx = torch.argmax(correlation)
    y_shift, x_shift = divmod(max_idx.item(), correlation.shape[1])

    # Calculate the shift relative to the center
    y_shift -= (img1.shape[0] - 1)
    x_shift -= (img1.shape[1] - 1)

    # Shift img2 to align with img1 using torch.roll (faster than numpy)
    aligned_img2 = torch.roll(img2_cuda, shifts=(y_shift, x_shift), dims=(0, 1))

    return aligned_img2.cpu().numpy(), (y_shift, x_shift)

def gaussian(x, amplitude, mean, sigma, offset=0):
    """Gaussian function for fitting with optional offset."""
    return amplitude * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset

def fit_gaussian_to_projection(image, axis=0, fit_offset=True, pixel_coords=None):
    """
    Fit Gaussian to image projection and return sigma value.
    
    Args:
        image: 2D numpy array
        axis: 0 for x-projection (sum along y), 1 for y-projection (sum along x)
        fit_offset: bool, whether to fit with baseline offset (default: True)
        pixel_coords: array, pixel coordinates (if None, uses pixel indices)
    
    Returns:
        sigma: fitted sigma value in same units as pixel_coords, None if fitting fails
    """
    # Get projection by summing along the specified axis
    projection = np.sum(image, axis=axis)
    
    # Create coordinate array for the projection
    if pixel_coords is not None:
        if axis == 0:
            # x-projection: use x coordinates
            x_proj = pixel_coords if len(pixel_coords) == image.shape[1] else pixel_coords
        else:
            # y-projection: use y coordinates  
            x_proj = pixel_coords if len(pixel_coords) == image.shape[0] else pixel_coords
    else:
        # Use pixel indices if no coordinates provided
        x_proj = np.arange(len(projection))
    
    # Check if projection has any signal
    if np.max(projection) == 0:
        return None
    
    # Initial guess for parameters
    amplitude_guess = np.max(projection) - np.min(projection)  # Peak above baseline
    mean_guess = np.average(x_proj, weights=projection)
    
    # Better sigma estimation using second moment
    variance = np.average((x_proj - mean_guess)**2, weights=projection)
    sigma_guess = np.sqrt(variance)
    
    # Ensure sigma_guess is reasonable
    coord_range = x_proj[-1] - x_proj[0]
    if sigma_guess < coord_range / 20:  # Too small
        sigma_guess = coord_range / 6.0
    elif sigma_guess > coord_range / 2:  # Too large
        sigma_guess = coord_range / 6.0
    
    try:
        if fit_offset:
            # Fit with offset (recommended for most cases)
            offset_guess = np.min(projection)
            p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]
            popt, _ = curve_fit(gaussian, x_proj, projection, 
                               p0=p0, maxfev=2000)
            return abs(popt[2])  # Return sigma (absolute value)
        else:
            # Fit without offset (for background-subtracted data)
            p0 = [amplitude_guess, mean_guess, sigma_guess]
            popt, _ = curve_fit(lambda x, a, m, s: gaussian(x, a, m, s, 0), 
                               x_proj, projection, 
                               p0=p0, maxfev=2000)
            return abs(popt[2])  # Return sigma (absolute value)
    except Exception as e:
        # If fitting fails, try a simpler approach
        try:
            # Try fitting without offset as fallback
            if fit_offset:
                # Subtract baseline and fit without offset
                baseline_proj = projection - np.min(projection)
                p0 = [amplitude_guess, mean_guess, sigma_guess]
                popt, _ = curve_fit(lambda x, a, m, s: gaussian(x, a, m, s, 0), 
                                   x_proj, baseline_proj, 
                                   p0=p0, maxfev=1000)
                return abs(popt[2])
        except:
            pass
        
        print(f"Gaussian fit failed for projection: {e}")
        return None
    
def calculate_rms_projection(image, axis=0, pixel_coords=None):
    """
    Calculate true RMS value of image projection.
    
    Args:
        image: 2D numpy array
        axis: 0 for x-projection (sum along y), 1 for y-projection (sum along x)
        pixel_coords: array, pixel coordinates (if None, uses pixel indices)
    
    Returns:
        rms: RMS value in same units as pixel_coords, None if calculation fails
    """
    # Get projection by summing along the specified axis
    projection = np.sum(image, axis=axis)
    
    # Create coordinate array for the projection
    if pixel_coords is not None:
        if axis == 0:
            # x-projection: use x coordinates
            x_proj = pixel_coords if len(pixel_coords) == image.shape[1] else pixel_coords
        else:
            # y-projection: use y coordinates  
            x_proj = pixel_coords if len(pixel_coords) == image.shape[0] else pixel_coords
    else:
        # Use pixel indices if no coordinates provided
        x_proj = np.arange(len(projection))
    
    # Check if projection has any signal
    if np.sum(projection) == 0:
        return None
    
    # Calculate weighted mean (centroid)
    x_mean = np.sum(x_proj * projection) / np.sum(projection)
    
    # Calculate true RMS: sqrt(sum((x - x_mean)^2 * rho(x)) / sum(rho(x)))
    numerator = np.sum((x_proj - x_mean)**2 * projection)
    denominator = np.sum(projection)
    
    rms = np.sqrt(numerator / denominator)
    
    return rms

def plot_original_predicted_difference(original, predicted, x_axis=None, y_axis=None, title_prefix=""):
    """
    Optimized plotting function with Gaussian fitting for sigma extraction.
    Adds sigma_x and sigma_y values from Gaussian fits to x and y projections.
    """

    original = normalize_images(original)
    predicted = normalize_images(predicted)

    # Convert to numpy if torch tensors
    if hasattr(original, "numpy"):
        original = original.cpu().numpy()
    if hasattr(predicted, "numpy"):
        predicted = predicted.detach().cpu().numpy()
    n_imgs = original.shape[0]

    px_size = 7.8e-6
    
    # Pre-calculate extent once if axes provided
    extent = None
    if x_axis is not None and y_axis is not None:
        extent = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
        print(extent)

    for i in range(n_imgs):
        orig_img = original[i]
        pred_img = predicted[i]

        # Use CUDA-optimized alignment (FFT-based, much faster)
        aligned_orig_img, shift = align_images_cross_correlation_cuda(pred_img, orig_img)

        # Calculate absolute difference
        diff_img = np.abs(aligned_orig_img - pred_img)

        # Fit Gaussians and extract sigma values
        # For original (aligned) image
        orig_sigma_x = fit_gaussian_to_projection(aligned_orig_img, axis=0)*px_size  # x-projection
        orig_sigma_y = fit_gaussian_to_projection(aligned_orig_img, axis=1)*px_size  # y-projection
        
        # For predicted image
        pred_sigma_x = fit_gaussian_to_projection(pred_img, axis=0)*px_size  # x-projection
        pred_sigma_y = fit_gaussian_to_projection(pred_img, axis=1)*px_size  # y-projection

        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot original (aligned)
        im0 = axs[0].imshow(aligned_orig_img, aspect='auto', extent=extent)
        axs[0].set_title(f"{title_prefix}Original (Aligned) [{i}]")
        plt.colorbar(im0, ax=axs[0], fraction=0.046)
        
        # Add sigma values as text on the plot
        if orig_sigma_x is not None and orig_sigma_y is not None:
            sigma_text = f'σₓ = {orig_sigma_x:.3f}\nσᵧ = {orig_sigma_y:.3f}'
            axs[0].text(0.02, 0.02, sigma_text, transform=axs[0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, verticalalignment='bottom')

        # Plot predicted
        im1 = axs[1].imshow(pred_img, aspect='auto', extent=extent)
        axs[1].set_title(f"{title_prefix}Predicted [{i}]")
        plt.colorbar(im1, ax=axs[1], fraction=0.046)
        
        # Add sigma values as text on the plot
        if pred_sigma_x is not None and pred_sigma_y is not None:
            sigma_text = f'σₓ = {pred_sigma_x:.3f}\nσᵧ = {pred_sigma_y:.3f}'
            axs[1].text(0.02, 0.02, sigma_text, transform=axs[1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, verticalalignment='bottom')

        # Plot difference (no sigma values needed)
        im2 = axs[2].imshow(diff_img, aspect='auto', extent=extent, cmap='hot')
        axs[2].set_title(f"{title_prefix}Abs. Difference [{i}]")
        plt.colorbar(im2, ax=axs[2], fraction=0.046)

        # Set labels
        for ax in axs:
            ax.set_xlabel('x(mm)')
            ax.set_ylabel('y(mm)')
        
        plt.tight_layout()
        plt.show()
        
        # Print sigma values to console as well
        print(f"Image [{i}] - Original: σₓ={orig_sigma_x:.4f}, σᵧ={orig_sigma_y:.4f}")
        print(f"Image [{i}] - Predicted: σₓ={pred_sigma_x:.4f}, σᵧ={pred_sigma_y:.4f}")
        print("-" * 50)

def unit_to_factor(unit):
    if unit == 'm':
        return 1
    elif unit == 'mm':
        return 1e3
    elif unit == 'um':
        return 1e6
    elif unit == 'nm':
        return 1e9
    else:
        raise ValueError(f"Unknown unit: {unit}")

class GaussFit:
    def __init__(self, x, y, fit_const=True):
        self.x = x
        self.y = y
        self.fit_const = fit_const
        self.popt, self.pcov = self.fit()
        self.reconstruction = self.gaussian(x, *self.popt)

    def gaussian(self, x, *p):
        A, mu, sigma = p[:3]
        y = A*np.exp(-(x-mu)**2/(2.*sigma**2))
        if self.fit_const:
            y = y + p[3]
        return y

    def fit(self):
        mean = self.x[np.argmax(self.y)]
        sigma = ((self.x - mean)**2 * self.y).sum() / self.y.sum()
        sigma = np.sqrt(sigma)
        p0 = [self.y.max(), mean, sigma]
        if self.fit_const:
            p0.append(0)
        try:
            popt, pcov = curve_fit(self.gaussian, self.x, self.y, p0=p0)
        except RuntimeError:
            print('Gauss fit failed')
            popt = p0
            pcov = np.zeros((len(p0), len(p0)))
        return popt, pcov

def prepare_images_for_plotting(generated_images, quad_k_values, lattice):
    """
    Prepare images for the plot_images_with_proj function.
    
    Parameters:
    -----------
    generated_images : list of torch.Tensor
        Images from generate_screen_image function
    quad_k_values : array-like
        Quadrupole strength values
    lattice : ScreenImageGenerator
        Your lattice instance to get coordinates
    
    Returns:
    --------
    list : List of tuples (image, x_coords, y_coords, quadk)
    """
    # Get coordinates from lattice
    x_coords = lattice.x_screen_coords.cpu().numpy()
    y_coords = lattice.y_screen_coords.cpu().numpy()
    
    # Create list of tuples
    images_for_plotting = []
    for i, image in enumerate(generated_images):
        # Convert tensor to numpy if needed
        if hasattr(image, 'cpu'):
            image_np = image.cpu().detach().numpy()
        else:
            image_np = image
            
        # Create tuple: (image, x_coords, y_coords, quadk)
        images_for_plotting.append((image_np, x_coords, y_coords, quad_k_values[i]))
    
    return images_for_plotting

def plot_images_with_proj(images, x_pixel_size, y_pixel_size, max_cols=4,
                           x_factor=None, y_factor=None, plot_proj=True, log=False,
                           loglog=False, revert_x=False, plot_gauss=True, slice_dict=None,
                           xlim=None, ylim=None, cmapname='hot', slice_cutoff=0,
                           gauss_color=('orange', 'orange'), proj_color=('green', 'green'),
                           slice_color='deepskyblue', slice_method='cut', plot_gauss_x=False,
                           plot_gauss_y=False, plot_proj_x=False, plot_proj_y=False,
                           gauss_alpha=None, cut_intensity_quantile=None, hlines=None,
                           hline_color='deepskyblue', vlines=None, vline_color='deepskyblue',
                           sqrt=False, plot_slice_lims=False):
    """
    Plots a list of images with projections using fit_gaussian_to_projection for Gaussian fitting.
    """
    n_images = len(images)
    if n_images == 0:
        print("No images to plot.")
        return

    # Determine subplot grid dimensions
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division

    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))

    # Flatten axes array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # Handle single subplot case
    else:
        axes = axes.flatten()

    # Use default factors if not provided
    if x_factor is None:
        x_factor = unit_to_factor('um')  # Assuming default unit is microns
    if y_factor is None:
        y_factor = unit_to_factor('um')  # Assuming default unit is microns

    # Plot each image
    outp_list = []
    for i, (image, x_coords, y_coords, quadk) in enumerate(images):
        ax = axes[i]

        # Determine axis limits based on xlim and ylim
        if xlim is None:
            index_x_min, index_x_max = 0, len(x_coords)
        else:
            index_x_min, index_x_max = sorted([np.argmin((x_coords - xlim[1])**2),
                                                 np.argmin((x_coords - xlim[0])**2)])
        if ylim is None:
            index_y_min, index_y_max = 0, len(y_coords)
        else:
            index_y_min, index_y_max = sorted([np.argmin((y_coords - ylim[1])**2),
                                                 np.argmin((y_coords - ylim[0])**2)])

        x_axis = x_coords[index_x_min:index_x_max]
        y_axis = y_coords[index_y_min:index_y_max]
        image = image[index_y_min:index_y_max, index_x_min:index_x_max]

        # Calculate extent for imshow
        extent = [x_axis[0] * x_factor, x_axis[-1] * x_factor,
                  y_axis[0] * y_factor, y_axis[-1] * y_factor]

        # Apply image transformations
        if cut_intensity_quantile:
            image = np.clip(image, 0, np.quantile(image, cut_intensity_quantile))

        if log:
            log_image = np.log(image + 1)
        elif loglog:
            log_image = np.log(np.log(image + 1) + 1)
        elif sqrt:
            log_image = np.sqrt(image)
        else:
            log_image = image

        # Plot image
        imshow_outp = ax.imshow(log_image, aspect='auto', extent=extent, origin='lower', cmap=plt.get_cmap(cmapname))

        # Add slice information
        if slice_dict is not None:
            old_lim = ax.get_xlim(), ax.get_ylim()
            mask = slice_dict['slice_current'] > slice_cutoff
            xx = slice_dict['slice_x'][mask] * x_factor
            yy = slice_dict[slice_method]['mean'][mask] * y_factor
            if 'eref' in slice_dict:
                yy = yy + slice_dict[slice_method]['eref'] * y_factor
            yy_err = np.sqrt(slice_dict[slice_method]['sigma_sq'][mask]) * y_factor
            ax.errorbar(xx, yy, yerr=yy_err, color=slice_color, marker='None', lw=1)

            if plot_slice_lims and slice_method in ('cut', 'rms'):
                lim1 = slice_dict[slice_method]['lim1'][mask] * y_factor
                lim2 = slice_dict[slice_method]['lim2'][mask] * y_factor
                if 'eref' in slice_dict:
                    lim1 = lim1 + slice_dict[slice_method]['eref'] * y_factor
                    lim2 = lim2 + slice_dict[slice_method]['eref'] * y_factor
                ax.plot(xx, lim1, color='yellow')
                ax.plot(xx, lim2, color='yellow')
            ax.set_xlim(*old_lim[0])
            ax.set_ylim(*old_lim[1])

        # Initialize variables for storing fit results
        sigma_x = sigma_y = None
        plot_proj_x_outp = None
        plot_gauss_x_outp = None

        # Plot x-projections
        if plot_proj or plot_proj_x:
            proj = image.sum(axis=-2)  # Sum along y-axis for x-projection
            proj_plot = (y_axis.min() + (y_axis.max() - y_axis.min()) * proj / proj.max() * 0.3) * y_factor
            plot_proj_x_outp = ax.plot(x_axis * x_factor, proj_plot, color=proj_color[0])
            
            if plot_gauss or plot_gauss_x:
                # Use fit_gaussian_to_projection to get sigma and create Gaussian curve
                sigma_x = fit_gaussian_to_projection(image, axis=0, pixel_coords=x_axis)
                
                if sigma_x is not None:
                    # Create Gaussian curve for visualization
                    x_proj_data = image.sum(axis=0)  # Raw projection data
                    
                    # Fit full Gaussian to get all parameters for plotting
                    try:
                        amplitude_guess = np.max(x_proj_data) - np.min(x_proj_data)
                        mean_guess = np.average(x_axis, weights=x_proj_data)
                        offset_guess = np.min(x_proj_data)
                        
                        popt, _ = curve_fit(gaussian, x_axis, x_proj_data, 
                                          p0=[amplitude_guess, mean_guess, sigma_x, offset_guess],
                                          maxfev=2000)
                        
                        # Scale the fitted Gaussian to match the projection plot scale
                        gauss_curve = gaussian(x_axis, *popt)
                        # Scale to match proj_plot range
                        gauss_scaled = (y_axis.min() + (y_axis.max() - y_axis.min()) * gauss_curve / gauss_curve.max() * 0.3) * y_factor
                        
                        plot_gauss_x_outp = ax.plot(x_axis * x_factor, gauss_scaled, 
                                                   color=gauss_color[0], alpha=gauss_alpha, linewidth=2)
                    except:
                        plot_gauss_x_outp = None

        plot_gauss_y_outp = None
        plot_proj_y_outp = None

        # Plot y-projections
        if plot_proj or plot_proj_y:
            proj = image.sum(axis=-1)  # Sum along x-axis for y-projection
            proj_plot = (x_axis.min() + (x_axis.max() - x_axis.min()) * proj / proj.max() * 0.3) * x_factor
            plot_proj_y_outp = ax.plot(proj_plot, y_axis * y_factor, color=proj_color[1])
            
            if plot_gauss or plot_gauss_y:
                # Use fit_gaussian_to_projection to get sigma and create Gaussian curve
                sigma_y = fit_gaussian_to_projection(image, axis=1, pixel_coords=y_axis)
                
                if sigma_y is not None:
                    # Create Gaussian curve for visualization
                    y_proj_data = image.sum(axis=1)  # Raw projection data
                    
                    # Fit full Gaussian to get all parameters for plotting
                    try:
                        amplitude_guess = np.max(y_proj_data) - np.min(y_proj_data)
                        mean_guess = np.average(y_axis, weights=y_proj_data)
                        offset_guess = np.min(y_proj_data)
                        
                        popt, _ = curve_fit(gaussian, y_axis, y_proj_data, 
                                          p0=[amplitude_guess, mean_guess, sigma_y, offset_guess],
                                          maxfev=2000)
                        
                        # Scale the fitted Gaussian to match the projection plot scale
                        gauss_curve = gaussian(y_axis, *popt)
                        # Scale to match proj_plot range
                        gauss_scaled = (x_axis.min() + (x_axis.max() - x_axis.min()) * gauss_curve / gauss_curve.max() * 0.3) * x_factor
                        
                        plot_gauss_y_outp = ax.plot(gauss_scaled, y_axis * y_factor, 
                                                   color=gauss_color[1], alpha=gauss_alpha, linewidth=2)
                    except:
                        plot_gauss_y_outp = None

        # Add horizontal and vertical lines
        if hlines is not None:
            for hline in hlines:
                ax.axhline(hline, color=hline_color)
        if vlines is not None:
            for vline in vlines:
                ax.axvline(vline, color=vline_color)

        # Revert x-axis if requested
        if revert_x:
            xlim = ax.get_xlim()
            ax.set_xlim(*xlim[::-1])

        # Set title and labels - include sigma values if available
        title = f"QuadK = {quadk:.3f}"
        if sigma_x is not None and sigma_y is not None:
            # Convert sigma to micrometers for display
            sigma_x_um = sigma_x * x_factor / unit_to_factor('um')
            sigma_y_um = sigma_y * y_factor / unit_to_factor('um')
            title += f"\nσₓ={sigma_x_um:.1f}μm, σᵧ={sigma_y_um:.1f}μm"
        
        ax.set_title(title)
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")

        # Add colorbar
        fig.colorbar(imshow_outp, ax=ax)

        # Store output - create dummy GaussFit-like objects for compatibility
        class FitResult:
            def __init__(self, sigma):
                self.sigma = sigma
                self.popt = [0, 0, sigma] if sigma is not None else None
                self.reconstruction = None
        
        outp = {
            'gf_x': FitResult(sigma_x),
            'gf_y': FitResult(sigma_y),
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'imshow_outp': imshow_outp,
            'plot_gauss_y_outp': plot_gauss_y_outp,
            'plot_proj_y_outp': plot_proj_y_outp,
            'plot_gauss_x_outp': plot_gauss_x_outp,
            'plot_proj_x_outp': plot_proj_x_outp,
            'extent': extent,
        }
        outp_list.append(outp)

    # Remove any unused subplots
    for i in range(n_images, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and show plot
    fig.tight_layout()
    plt.show()

    return outp_list

def plot_sigmas_scatter(original, predicted, px_size):
    """
    For each image in original and predicted, compute sigma_x and sigma_y,
    and plot them with sigma_x on the x-axis and sigma_y on the y-axis.
    """
    # Ensure numpy arrays
    if hasattr(original, "numpy"):
        original = original.cpu().numpy()
    if hasattr(predicted, "numpy"):
        predicted = predicted.detach().cpu().numpy()
    
    n_imgs = original.shape[0]
    orig_sigmas_x, orig_sigmas_y = [], []
    pred_sigmas_x, pred_sigmas_y = [], []
    
    for i in range(n_imgs):
        orig_img = original[i]
        pred_img = predicted[i]

        orig_img, shift = align_images_cross_correlation_cuda(pred_img, orig_img)
        
        # Compute sigmas for original image
        orig_sigma_x = fit_gaussian_to_projection(orig_img, axis=0)*px_size*1e3
        orig_sigma_y = fit_gaussian_to_projection(orig_img, axis=1)*px_size*1e3
        # Compute sigmas for predicted image
        pred_sigma_x = fit_gaussian_to_projection(pred_img, axis=0)*px_size*1e3
        pred_sigma_y = fit_gaussian_to_projection(pred_img, axis=1)*px_size*1e3
        
        if orig_sigma_x is not None and orig_sigma_y is not None:
            orig_sigmas_x.append(orig_sigma_x)
            orig_sigmas_y.append(orig_sigma_y)
        if pred_sigma_x is not None and pred_sigma_y is not None:
            pred_sigmas_x.append(pred_sigma_x)
            pred_sigmas_y.append(pred_sigma_y)
    
    return orig_sigmas_x, orig_sigmas_y, pred_sigmas_x, pred_sigmas_y
