import argparse
import sys
import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# Add required paths to sys.path
for path in ['/afs/psi.ch/intranet/SF/Beamdynamics/Harsh/ps_reconstruction/', 
             '/afs/psi.ch/intranet/SF/Beamdynamics/Harsh/']:
    if path not in sys.path:
        sys.path.append(path)

from ps_reconstruction.core.image_processing import crop_images_to_bright_spot_centers
from ps_reconstruction.core.dataloader import BeamImageDataset, split_dataset
from ps_reconstruction.core.simulation import ScreenImageGenerator
from ps_reconstruction.core.losses import normalize_images
from ps_reconstruction.utils.visualization import fit_gaussian_to_projection, calculate_rms_projection


def load_and_process_data(dataset_num, crop_size, to_use_indices):
    """Load and process HDF5 data based on dataset number"""
    
    # Select dataset file
    if dataset_num == 1:
        dfile = '/sf/data/measurements/2025/05/22/20250522_212838_EmittanceTool.h5'
    elif dataset_num == 2:
        dfile = '/sf/data/measurements/2025/06/30/20250630_193506_EmittanceTool.h5'
    else:
        raise ValueError("Dataset must be 1 or 2")
    
    # Load HDF5 file
    with h5py.File(dfile, 'r') as f:
        original_r11 = f['/Input/R11'][()]
        original_r12 = f['/Input/R12'][()]
        original_r33 = f['/Input/R33'][()]
        original_r34 = f['/Input/R34'][()]
        x_phase_design = f['/Meta_data/proj_emittance_X/phase_design'][()]
        y_phase_design = f['/Meta_data/proj_emittance_Y/phase_design'][()]
        original_quad_k = f['/Input/QuadK'][()][-1]
        images = f['/Raw_data/image'][()]
        x_axis = f['/Raw_data/x_axis'][0,0].astype(float)/1e6
        y_axis = f['/Raw_data/y_axis'][0,0].astype(float)/1e6
    
    # Apply background subtraction
    proc_images = np.array(images, dtype=np.float64) - 1000
    proc_images[proc_images < 0] = 0
    
    # Crop images to bright spot centers
    processed_images_array, processed_x_axis, processed_y_axis = crop_images_to_bright_spot_centers(
        proc_images[:, 0],
        target_x=crop_size,
        target_y=crop_size,
        x_axis=x_axis,
        y_axis=y_axis
    )
    
    # Filter all data by to_use_indices
    filtered_r11 = original_r11[to_use_indices]
    filtered_r12 = original_r12[to_use_indices]
    filtered_r33 = original_r33[to_use_indices]
    filtered_r34 = original_r34[to_use_indices]
    filtered_quad_k = original_quad_k[to_use_indices]
    filtered_x_phase_design = x_phase_design[to_use_indices]
    filtered_y_phase_design = y_phase_design[::-1][to_use_indices]
    
    processed_tensors = torch.tensor(processed_images_array[to_use_indices], dtype=torch.float32)
    
    # Calculate pixel sizes
    x_pixel_size = abs(processed_x_axis[0][1] - processed_x_axis[0][0])
    y_pixel_size = abs(processed_y_axis[0][1] - processed_y_axis[0][0])
    
    return {
        'processed_tensors': processed_tensors,
        'filtered_r11': filtered_r11,
        'filtered_r12': filtered_r12,
        'filtered_r33': filtered_r33,
        'filtered_r34': filtered_r34,
        'filtered_quad_k': filtered_quad_k,
        'filtered_x_phase_design': filtered_x_phase_design,
        'filtered_y_phase_design': filtered_y_phase_design,
        'x_pixel_size': x_pixel_size,
        'y_pixel_size': y_pixel_size
    }


def calculate_beam_metrics(generated_images, original_images, x_pixel_size, y_pixel_size, n_phases):
    """Calculate sigma and RMS values for beam images"""
    
    normalized_generated = normalize_images(generated_images).cpu().numpy()
    normalized_original = normalize_images(original_images).cpu().numpy()
    
    orig_sigmas_x, orig_sigmas_y = [], []
    pred_sigmas_x, pred_sigmas_y = [], []
    orig_rms_x, orig_rms_y = [], []
    pred_rms_x, pred_rms_y = [], []
    
    for i in range(n_phases):
        generated_img = normalized_generated[i]
        orig_img = normalized_original[i]
        
        # Calculate sigma values
        orig_sigmas_x.append(fit_gaussian_to_projection(orig_img, axis=0) * x_pixel_size * 1e6)
        orig_sigmas_y.append(fit_gaussian_to_projection(orig_img, axis=1) * y_pixel_size * 1e6)
        pred_sigmas_x.append(fit_gaussian_to_projection(generated_img, axis=0) * x_pixel_size * 1e6)
        pred_sigmas_y.append(fit_gaussian_to_projection(generated_img, axis=1) * y_pixel_size * 1e6)
        
        # Calculate RMS values
        orig_rms_x.append(calculate_rms_projection(orig_img, axis=0) * x_pixel_size * 1e6)
        orig_rms_y.append(calculate_rms_projection(orig_img, axis=1) * y_pixel_size * 1e6)
        pred_rms_x.append(calculate_rms_projection(generated_img, axis=0) * x_pixel_size * 1e6)
        pred_rms_y.append(calculate_rms_projection(generated_img, axis=1) * y_pixel_size * 1e6)
    
    return {
        'orig_sigmas_x': orig_sigmas_x, 'orig_sigmas_y': orig_sigmas_y,
        'pred_sigmas_x': pred_sigmas_x, 'pred_sigmas_y': pred_sigmas_y,
        'orig_rms_x': orig_rms_x, 'orig_rms_y': orig_rms_y,
        'pred_rms_x': pred_rms_x, 'pred_rms_y': pred_rms_y
    }


def create_two_line_comparison(x_values, orig_x, orig_y, pred_x, pred_y, 
                              label='Sigma', x_label='Phase', title_suffix=''):
    """Create comparison plot with two lines (original vs predicted)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: X values
    axes[0].plot(x_values, orig_x, c='green', 
                label=f'Original image {label} X', marker='o')
    axes[0].plot(x_values, pred_x, c='blue', 
                label=f'Predicted {label} X', marker='o')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(f'{label} X (µm)')
    axes[0].set_title(f'{label} X Values vs {x_label}')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right subplot: Y values
    axes[1].plot(x_values, orig_y, c='green', 
                label=f'Original image {label} Y', marker='o')
    axes[1].plot(x_values, pred_y, c='blue', 
                label=f'Predicted {label} Y', marker='o')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(f'{label} Y (µm)')
    axes[1].set_title(f'{label} Y Values vs {x_label}')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.suptitle(f'{label} Beam Size Comparison{title_suffix}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def create_three_line_comparison(x_values, orig_x, orig_y, pred1_x, pred1_y, 
                                pred2_x, pred2_y, label='Sigma', x_label='Phase', title_suffix=''):
    """Create comparison plot with three lines (original vs two predictions)"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: X values
    axes[0].plot(x_values, orig_x, c='red', linestyle='--', 
                label=f'Original image {label} X', marker='o')
    axes[0].plot(x_values, pred1_x, c='blue', 
                label='Version 1 Prediction', marker='o')
    axes[0].plot(x_values, pred2_x, c='green', 
                label='Version 2 Prediction', marker='o')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(f'{label} X (µm)')
    axes[0].set_title(f'{label} X Values vs {x_label}')
    axes[0].legend()
    axes[0].grid(True)
    
    # Right subplot: Y values
    axes[1].plot(x_values, orig_y, c='red', linestyle='--', 
                label=f'Original image {label} Y', marker='o')
    axes[1].plot(x_values, pred1_y, c='blue', 
                label='Version 1 Prediction', marker='o')
    axes[1].plot(x_values, pred2_y, c='green', 
                label='Version 2 Prediction', marker='o')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(f'{label} Y (µm)')
    axes[1].set_title(f'{label} Y Values vs {x_label}')
    axes[1].legend()
    axes[1].grid(True)
    
    fig.suptitle(f'{label} Beam Size Comparison{title_suffix}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate beam size comparison plots')
    parser.add_argument('--dataset', type=int, choices=[1, 2], default=2,
                       help='Dataset number (1 or 2)')
    parser.add_argument('--crop_size', type=int, default=160,
                       help='Crop size for images (default: 160)')
    parser.add_argument('--v1', type=str, required=True,
                       help='Version 1 model path')
    parser.add_argument('--v2', type=str, default=None,
                       help='Version 2 model path (optional)')
    parser.add_argument('--folder_path', type=str, default='./logs/comparison_plots',
                       help='Folder path to save plots (default: ./logs)')
    
    args = parser.parse_args()
    
    # Create output folder if it doesn't exist
    output_folder = Path(args.folder_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Set dataset-specific parameters
    if args.dataset == 1:
        to_use_indices = [3,4,5,6,7,9,13,14,15,16]
        to_use_indices.extend(list(int(i) for i in np.arange(20,30,1)))
        n_phases = 20
        dataset_name = 'd1'
    else:  # dataset == 2
        to_use_indices = np.arange(0,31,1)
        n_phases = 31
        dataset_name = 'd2'
    
    print(f"Loading and processing dataset {args.dataset}...")
    data = load_and_process_data(args.dataset, args.crop_size, to_use_indices)
    
    # Load trained beam models
    print(f"Loading beam model from {args.v1}...")
    gt_beam1 = torch.load(f'/afs/psi.ch/intranet/SF/Beamdynamics/Harsh/ps_reconstruction/examples/logs/beam_training/version_{args.v1}/final_results/reconstructed_beam.pt', weights_only=False)
    
    # Create lattice simulator
    lattice = ScreenImageGenerator(
        data['filtered_r11'], data['filtered_r12'], 
        data['filtered_r33'], data['filtered_r34'], 
        data['filtered_quad_k'], args.crop_size, args.crop_size, 
        7.7e-6, 7.8e-6, device='cuda'
    )
    
    # Generate images from first model
    print("Generating images from first model...")
    generated_images1 = lattice.generate_screen_image(gt_beam1, gaussian_blur_sigma=3)
    
    # Calculate metrics for first model
    metrics1 = calculate_beam_metrics(
        generated_images1, data['processed_tensors'],
        data['x_pixel_size'], data['y_pixel_size'], n_phases
    )
    
    # Use phase design for x-axis
    x_axis_values = data['filtered_x_phase_design']
    
    if args.v2:
        print(f"Loading second beam model from {args.v2}...")
        gt_beam2 = torch.load(f'/afs/psi.ch/intranet/SF/Beamdynamics/Harsh/ps_reconstruction/examples/logs/beam_training/version_{args.v2}/final_results/reconstructed_beam.pt', weights_only=False)
        
        print("Generating images from second model...")
        generated_images2 = lattice.generate_screen_image(gt_beam2, gaussian_blur_sigma=3)
        
        # Calculate metrics for second model
        metrics2 = calculate_beam_metrics(
            generated_images2, data['processed_tensors'],
            data['x_pixel_size'], data['y_pixel_size'], n_phases
        )
        
        # Create three-line comparison plots
        print("Creating three-line comparison plots...")
        
        # Sigma comparison
        sigma_fig = create_three_line_comparison(
            x_axis_values,
            metrics1['orig_sigmas_x'], metrics1['orig_sigmas_y'],
            metrics1['pred_sigmas_x'], metrics1['pred_sigmas_y'],
            metrics2['pred_sigmas_x'], metrics2['pred_sigmas_y'],
            label='Sigma', x_label='Phase'
        )
        
        # RMS comparison
        rms_fig = create_three_line_comparison(
            x_axis_values,
            metrics1['orig_rms_x'], metrics1['orig_rms_y'],
            metrics1['pred_rms_x'], metrics1['pred_rms_y'],
            metrics2['pred_rms_x'], metrics2['pred_rms_y'],
            label='RMS', x_label='Phase'
        )
        
    else:
        # Create two-line comparison plots
        print("Creating two-line comparison plots...")
        
        # Sigma comparison
        sigma_fig = create_two_line_comparison(
            x_axis_values,
            metrics1['orig_sigmas_x'], metrics1['orig_sigmas_y'],
            metrics1['pred_sigmas_x'], metrics1['pred_sigmas_y'],
            label='Sigma', x_label='Phase'
        )
        
        # RMS comparison
        rms_fig = create_two_line_comparison(
            x_axis_values,
            metrics1['orig_rms_x'], metrics1['orig_rms_y'],
            metrics1['pred_rms_x'], metrics1['pred_rms_y'],
            label='RMS', x_label='Phase'
        )
    
    if args.v2:
        # Save plots
        sigma_path = output_folder / f'sigma_comparison_{dataset_name}_v{args.v1}_v{args.v2}.png'
        rms_path = output_folder / f'rms_comparison_{dataset_name}_v{args.v1}_v{args.v2}.png'
    else:
        # Save plots
        sigma_path = output_folder / f'sigma_comparison_{dataset_name}_v{args.v1}.png'
        rms_path = output_folder / f'rms_comparison_{dataset_name}_v{args.v1}.png'
    
    print(f"Saving plots to {output_folder}...")
    sigma_fig.savefig(sigma_path, dpi=300, bbox_inches='tight')
    rms_fig.savefig(rms_path, dpi=300, bbox_inches='tight')
    
    print(f"Plots saved:")
    print(f"  - Sigma comparison: {sigma_path}")
    print(f"  - RMS comparison: {rms_path}")
    
    plt.close(sigma_fig)
    plt.close(rms_fig)
    
    print("Done!")


if __name__ == "__main__":
    main()