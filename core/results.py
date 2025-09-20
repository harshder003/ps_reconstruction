import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from .losses import normalize_images
from ..utils.visualization import align_images_cross_correlation_cuda, fit_gaussian_to_projection

# Function to save all results to the logs folder
def save_results_to_logs(reconstructed_beam, target_images_test, predicted_images_test, 
                        quad_k_test, emittance_results, logger_log_dir):
    """
    Save all results to the logs folder structure
    
    Parameters:
    -----------
    reconstructed_beam : ParticleBeam
        The reconstructed beam from the generator
    target_images_test : torch.Tensor
        Original test images
    predicted_images_test : torch.Tensor
        Predicted test images
    quad_k_test : torch.Tensor
        Quadrupole K values for test set
    emittance_results : dict
        Dictionary containing all emittance, alpha, beta values
    logger_log_dir : str
        Path to the logger directory (e.g., logs/beam_training/version_X)
    """
    
    # Create results directory
    results_dir = os.path.join(logger_log_dir, "final_results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving results to: {results_dir}")
    
    # 1. Save reconstructed beam as .pt file
    beam_save_path = os.path.join(results_dir, "reconstructed_beam.pt")
    torch.save(reconstructed_beam, beam_save_path)
    print(f"✓ Saved reconstructed beam to: {beam_save_path}")
    
    # 2. Save beam distribution plot
    plt.figure(figsize=(12, 10))
    try:
        reconstructed_beam.plot_distribution()
        plt.suptitle("Reconstructed Beam Distribution", fontsize=16)
        beam_plot_path = os.path.join(results_dir, "reconstructed_beam_distribution.png")
        plt.savefig(beam_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved beam distribution plot to: {beam_plot_path}")
    except Exception as e:
        print(f"Warning: Could not save beam distribution plot: {e}")
    
    # 3. Save emittance results as text file and .pt file
    emittance_txt_path = os.path.join(results_dir, "emittance_results.txt")
    
    with open(emittance_txt_path, 'w') as f:
        f.write("RECONSTRUCTED BEAM ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Reconstructed parameters
        f.write("RECONSTRUCTED PARAMETERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Emittance x: {emittance_results['emittance_x']:.4e} m.rad\n")
        f.write(f"Alpha x: {emittance_results['alpha_x']:.4f}\n")
        f.write(f"Beta x: {emittance_results['beta_x']:.4f} m\n\n")
        f.write(f"Emittance y: {emittance_results['emittance_y']:.4e} m.rad\n")
        f.write(f"Alpha y: {emittance_results['alpha_y']:.4f}\n")
        f.write(f"Beta y: {emittance_results['beta_y']:.4f} m\n\n")
        
        # Target parameters
        f.write("FROM TOOL PARAMETERS:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Original Emittance x: {emittance_results['target_emittance_x']:.4e} m.rad\n")
        f.write(f"Original Alpha x: {emittance_results['target_alpha_x']:.4f}\n")
        f.write(f"Original Beta x: {emittance_results['target_beta_x']:.4f} m\n\n")
        f.write(f"Original Emittance y: {emittance_results['target_emittance_y']:.4e} m.rad\n")
        f.write(f"Original Alpha y: {emittance_results['target_alpha_y']:.4f}\n")
        f.write(f"Original Beta y: {emittance_results['target_beta_y']:.4f} m\n\n")
        
        # Error analysis
        f.write("ERROR ANALYSIS:\n")
        f.write("-" * 15 + "\n")
        emittance_x_error = abs(emittance_results['emittance_x'] - emittance_results['target_emittance_x'])
        emittance_y_error = abs(emittance_results['emittance_y'] - emittance_results['target_emittance_y'])
        alpha_x_error = abs(emittance_results['alpha_x'] - emittance_results['target_alpha_x'])
        alpha_y_error = abs(emittance_results['alpha_y'] - emittance_results['target_alpha_y'])
        beta_x_error = abs(emittance_results['beta_x'] - emittance_results['target_beta_x'])
        beta_y_error = abs(emittance_results['beta_y'] - emittance_results['target_beta_y'])
        
        f.write(f"Emittance x error: {emittance_x_error}m.rad\n")
        f.write(f"Emittance y error: {emittance_y_error}m.rad\n")
        f.write(f"Alpha x error: {alpha_x_error}\n")
        f.write(f"Alpha y error: {alpha_y_error}\n")
        f.write(f"Beta x error: {beta_x_error}\n")
        f.write(f"Beta y error: {beta_y_error}\n")
    
    print(f"✓ Saved emittance results to: {emittance_txt_path}")
    
    # 4. Generate sigma comparison plots and CSV
    print("Creating sigma comparison plots and CSV...")
    
    # Convert tensors to numpy for processing
    target_np = normalize_images(target_images_test).cpu().numpy()
    predicted_np = normalize_images(predicted_images_test).cpu().numpy()
    quad_k_np = quad_k_test.cpu().numpy() if hasattr(quad_k_test, 'cpu') else quad_k_test
    
    # Pixel size for sigma calculations
    px_size = 7.8e-6
    
    # Calculate sigma values
    n_images = target_np.shape[0]
    orig_sigmas_x, orig_sigmas_y = [], []
    pred_sigmas_x, pred_sigmas_y = [], []
    
    for i in range(n_images):
        target_img = target_np[i]
        pred_img = predicted_np[i]
        
        # Align images
        aligned_target_img, shift = align_images_cross_correlation_cuda(pred_img, target_img)
        
        # Ensure images are numpy arrays after alignment
        if hasattr(aligned_target_img, "numpy"):
            aligned_target_img = aligned_target_img.cpu().numpy()
        if hasattr(pred_img, "numpy"):
            pred_img = pred_img.cpu().numpy()
        
        # Calculate sigma values for original (aligned) image
        orig_sigma_x = fit_gaussian_to_projection(aligned_target_img, axis=0) * px_size
        orig_sigma_y = fit_gaussian_to_projection(aligned_target_img, axis=1) * px_size
        
        # Calculate sigma values for predicted image
        pred_sigma_x = fit_gaussian_to_projection(pred_img, axis=0) * px_size
        pred_sigma_y = fit_gaussian_to_projection(pred_img, axis=1) * px_size
        
        if orig_sigma_x is not None and orig_sigma_y is not None:
            orig_sigmas_x.append(orig_sigma_x)
            orig_sigmas_y.append(orig_sigma_y)
        else:
            orig_sigmas_x.append(np.nan)
            orig_sigmas_y.append(np.nan)
            
        if pred_sigma_x is not None and pred_sigma_y is not None:
            pred_sigmas_x.append(pred_sigma_x)
            pred_sigmas_y.append(pred_sigma_y)
        else:
            pred_sigmas_x.append(np.nan)
            pred_sigmas_y.append(np.nan)
    
    # Create sigma comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: Sigma X
    axes[0].plot(quad_k_np, orig_sigmas_x, c='blue', linestyle='--', label='Original Sigma X')
    axes[0].plot(quad_k_np, pred_sigmas_x, c='blue', label='Predicted Sigma X')
    axes[0].set_xlabel('Quad K')
    axes[0].set_ylabel('Sigma X (m)')
    axes[0].set_title('Sigma X Values vs Quad K')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right subplot: Sigma Y
    axes[1].plot(quad_k_np, orig_sigmas_y, c='red', linestyle='--', label='Original Sigma Y')
    axes[1].plot(quad_k_np, pred_sigmas_y, c='red', label='Predicted Sigma Y')
    axes[1].set_xlabel('Quad K')
    axes[1].set_ylabel('Sigma Y (m)')
    axes[1].set_title('Sigma Y Values vs Quad K')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.suptitle('Sigma Comparison: Original vs Predicted')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save sigma comparison plot
    sigma_plot_path = os.path.join(results_dir, "sigma_comparison.png")
    plt.savefig(sigma_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved sigma comparison plot to: {sigma_plot_path}")
    
    # Save sigma values to CSV
    sigma_data = {
        'quad_k': quad_k_np,
        'orig_sigma_x': orig_sigmas_x,
        'orig_sigma_y': orig_sigmas_y,
        'pred_sigma_x': pred_sigmas_x,
        'pred_sigma_y': pred_sigmas_y
    }
    
    df = pd.DataFrame(sigma_data)
    csv_path = os.path.join(results_dir, "sigma_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved sigma values to CSV: {csv_path}")
    
    # 5. Create and save concatenated comparison images with proper axes
    print("Creating concatenated comparison images with proper axes...")

    # Convert tensors to numpy for plotting
    target_np = normalize_images(target_images_test).cpu().numpy()
    predicted_np = normalize_images(predicted_images_test).cpu().numpy()

    # Use all available images
    n_images = target_np.shape[0]
    px_size_display = 7.8e-6  # Pixel size
    
    # Layout configuration - arrange images in columns, each image takes 3 subplot columns
    max_images_per_col = 5  # Maximum images per column
    n_image_cols = (n_images + max_images_per_col - 1) // max_images_per_col  # Ceiling division
    n_rows = min(n_images, max_images_per_col)
    n_cols = n_image_cols * 3  # Each image needs 3 subplot columns (original, predicted, difference)
    
    # Create coordinate arrays based on image dimensions
    img_height, img_width = target_np.shape[1], target_np.shape[2]
    
    # Create x and y coordinate arrays in meters, then convert to microns
    x_coords = np.linspace(-img_width/2 * px_size_display, img_width/2 * px_size_display, img_width) * 1e6  # Convert to μm
    y_coords = np.linspace(-img_height/2 * px_size_display, img_height/2 * px_size_display, img_height) * 1e6  # Convert to μm
    
    # Calculate extent for imshow
    extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]

    # Create figure with dynamic size
    fig_width = 5 * n_image_cols  # Scale width based on number of image columns
    fig_height = 3.2 * n_rows     # Scale height based on number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Ensure axes is 2D even if there's only one row or column
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_images):
        # Determine position in grid
        image_col = i // max_images_per_col  # Which image column (0, 1, 2, ...)
        row = i % max_images_per_col         # Which row within that image column
        col_offset = image_col * 3           # Starting subplot column for this image column
        
        target_img = target_np[i]
        pred_img = predicted_np[i]

        # Align images
        aligned_target_img, shift = align_images_cross_correlation_cuda(pred_img, target_img)
        diff_img = np.abs(aligned_target_img - pred_img)

        # Fit Gaussians and extract sigma values
        # For original (aligned) image
        orig_sigma_x = fit_gaussian_to_projection(aligned_target_img, axis=0) * px_size_display * 1e6  # Convert to μm
        orig_sigma_y = fit_gaussian_to_projection(aligned_target_img, axis=1) * px_size_display * 1e6  # Convert to μm

        # For predicted image
        pred_sigma_x = fit_gaussian_to_projection(pred_img, axis=0) * px_size_display * 1e6  # Convert to μm
        pred_sigma_y = fit_gaussian_to_projection(pred_img, axis=1) * px_size_display * 1e6  # Convert to μm

        # Reference image
        ax_orig = axes[row, col_offset]
        im1 = ax_orig.imshow(aligned_target_img, cmap='hot', origin='lower', extent=extent, aspect='equal')
        ax_orig.set_title(f'Reference {i+1}', fontsize=10)
        ax_orig.set_xlabel('x (μm)', fontsize=8)
        ax_orig.set_ylabel('y (μm)', fontsize=8)
        ax_orig.tick_params(axis='both', which='major', labelsize=8)
        cbar = plt.colorbar(im1, ax=ax_orig, fraction=0.03, pad=0.02, shrink=0.9)
        cbar.ax.tick_params(labelsize=6)

        # Add sigma values as text on the plot
        if orig_sigma_x is not None and orig_sigma_y is not None:
            sigma_text = f'σₓ={orig_sigma_x:.1f}μm\nσᵧ={orig_sigma_y:.1f}μm'
            ax_orig.text(0.02, 0.02, sigma_text, transform=ax_orig.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=7, verticalalignment='bottom')

        # Predicted image
        ax_pred = axes[row, col_offset + 1]
        im2 = ax_pred.imshow(pred_img, cmap='hot', origin='lower', extent=extent, aspect='equal')
        ax_pred.set_title(f'Predicted {i+1}', fontsize=10)
        ax_pred.set_xlabel('x (μm)', fontsize=8)
        ax_pred.set_ylabel('y (μm)', fontsize=8)
        ax_pred.tick_params(axis='both', which='major', labelsize=8)
        cbar = plt.colorbar(im2, ax=ax_pred, fraction=0.03, pad=0.02, shrink=0.9)
        cbar.ax.tick_params(labelsize=6)

        # Add sigma values as text on the plot
        if pred_sigma_x is not None and pred_sigma_y is not None:
            sigma_text = f'σₓ={pred_sigma_x:.1f}μm\nσᵧ={pred_sigma_y:.1f}μm'
            ax_pred.text(0.02, 0.02, sigma_text, transform=ax_pred.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                       fontsize=7, verticalalignment='bottom')

        # Difference image
        ax_diff = axes[row, col_offset + 2]
        im3 = ax_diff.imshow(diff_img, cmap='RdBu_r', origin='lower', extent=extent, aspect='equal')
        ax_diff.set_title(f'Difference {i+1}', fontsize=10)
        ax_diff.set_xlabel('x (μm)', fontsize=8)
        ax_diff.set_ylabel('y (μm)', fontsize=8)
        ax_diff.tick_params(axis='both', which='major', labelsize=8)
        cbar = plt.colorbar(im3, ax=ax_diff, fraction=0.03, pad=0.02, shrink=0.8)
        cbar.ax.tick_params(labelsize=6)

    plt.suptitle(f"Reference vs Predicted vs Difference Images (Test Set - {n_images} images)", fontsize=16)
    plt.subplots_adjust(wspace=0.45, hspace=0.01, left=0.05, right=0.95, top=0.95, bottom=0.05)

    comparison_plot_path = os.path.join(results_dir, "test_images_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison images to: {comparison_plot_path}")
    
    return results_dir