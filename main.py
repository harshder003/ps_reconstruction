import argparse
import gc
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
import os

# Add necessary paths to sys.path
for path in ['/afs/psi.ch/intranet/SF/Beamdynamics/Harsh/']:
    if path not in sys.path:
        sys.path.append(path)

# Import custom modules
from ps_reconstruction.core.image_processing import crop_images_to_bright_spot_centers
from ps_reconstruction.core.dataloader import BeamImageDataset, create_beam_dataloader, beam_collate_fn
from ps_reconstruction.core.simulation import ScreenImageGenerator
from ps_reconstruction.core.beam_generator import ParticleTransformer, NNParticleBeamGenerator, NNTransform
from ps_reconstruction.core.train import train_beam_generator_with_dataloader
from ps_reconstruction.core.results import save_results_to_logs

# Dynamically determine the version number
def get_next_version(log_dir: str) -> str:
    """
    Determines the next version number by checking existing folders in the log directory.
    """
    if not os.path.exists(log_dir):
        return "version_1"
    existing_versions = [
        int(folder_name.replace("version_", ""))
        for folder_name in os.listdir(log_dir)
        if folder_name.startswith("version_") and folder_name.replace("version_", "").isdigit()
    ]
    next_version = max(existing_versions, default=0) + 1
    return f"version_{next_version}"

if __name__ == "__main__":
    # Collect garbage
    gc.collect()

    # Argument parser for version and dataset selection
    parser = argparse.ArgumentParser(description="Beam Dynamics Reconstruction")
    parser.add_argument("--version", type=str, help="Version for logging")
    parser.add_argument("--dataset", type=int, choices=[1, 2], required=True, help="Dataset to use: 1 or 2")
    args = parser.parse_args()

    # Define the log directory
    log_dir = './logs/beam_training'

    # Determine version dynamically if not provided
    version = args.version if args.version else get_next_version(log_dir)

    # Define dataset paths and parameters
    dataset_paths = {
        1: '/sf/data/measurements/2025/05/22/20250522_212838_EmittanceTool.h5',
        2: '/sf/data/measurements/2025/06/30/20250630_193506_EmittanceTool.h5'
    }
    crop_params = {
        1: {"target_x": 120, "target_y": 120, "to_use_indices": [3, 4, 5, 6, 7, 9, 13, 14, 15, 16] + list(range(20, 30))},
        2: {"target_x": 160, "target_y": 160, "to_use_indices": np.arange(0, 31, 1)}
    }

    # Load the selected dataset
    dfile = dataset_paths[args.dataset]
    crop_config = crop_params[args.dataset]
    to_use_indices = crop_config["to_use_indices"]

    with h5py.File(dfile, 'r') as f:
        # Store original data (before slicing) for ScreenImageGenerator __init__
        original_r11 = f['/Input/R11'][()]
        original_r12 = f['/Input/R12'][()]
        original_r33 = f['/Input/R33'][()]
        original_r34 = f['/Input/R34'][()]
        original_quad_k = f['/Input/QuadK'][()] # shape: (n_phases,)
        x_emittance = f['/Meta_data/proj_emittance_X/emittance'][()]
        x_beta = f['/Meta_data/proj_emittance_X/beta'][()]
        x_alpha = f['/Meta_data/proj_emittance_X/alpha'][()]
        y_emittance = f['/Meta_data/proj_emittance_Y/emittance'][()]
        y_beta = f['/Meta_data/proj_emittance_Y/beta'][()]
        y_alpha = f['/Meta_data/proj_emittance_Y/alpha'][()]
        p0c = f['/Input/energy_eV'][()]
        images = f['/Raw_data/image'][()]  # shape: (n_phases, n_images, y, x)
        x_axis = f['/Raw_data/x_axis'][0,0].astype(float)/1e6  # shape: (x,)
        y_axis = f['/Raw_data/y_axis'][0,0].astype(float)/1e6  # shape: (y,)
        screen_res = f['/Input/screen_resolution'][()] # e.g., (100, 100)

    # Apply background subtraction
    proc_images = np.array(images, dtype=np.float64) - 1000
    proc_images[proc_images < 0 ] = 0

    # Crop images to bright spot centers
    processed_images_array, processed_x_axis, processed_y_axis = crop_images_to_bright_spot_centers(
        proc_images[:, 0],  # Assuming you always use the first image from n_images dimension
        target_x=crop_config["target_x"],
        target_y=crop_config["target_y"],
        x_axis=x_axis,
        y_axis=y_axis
    )

    # Filter all data by to_use_indices
    filtered_r11 = original_r11[to_use_indices]
    filtered_r12 = original_r12[to_use_indices]
    filtered_r33 = original_r33[to_use_indices]
    filtered_r34 = original_r34[to_use_indices]
    filtered_quad_k = original_quad_k[-1][to_use_indices]

    processed_tensors = torch.tensor(processed_images_array[to_use_indices], dtype=torch.float32)

    # Ensure R-matrix and quad_k are 1D arrays of values for direct mapping to each sample
    # (The DataLoader will stack them later)
    quad_torch_k_for_dataset = torch.tensor(filtered_quad_k, dtype=torch.float32).flatten() # Ensure 1D

    # Calculate pixel sizes for the cropped images
    x_pixel_size = abs(processed_x_axis[0][1] - processed_x_axis[0][0])
    y_pixel_size = abs(processed_y_axis[0][1] - processed_y_axis[0][0])
    x_pixel_size_tensor = torch.tensor(x_pixel_size, dtype=torch.float32)
    y_pixel_size_tensor = torch.tensor(y_pixel_size, dtype=torch.float32)

    # Function to split dataset (re-include it here for self-contained cell)
    def split_dataset(dataset: BeamImageDataset, train_indices: list) -> tuple[Subset, Subset]:
        """
        Splits a BeamImageDataset into training and testing subsets based on provided indices.
        """
        train_indices_set = set(train_indices)
        all_indices = set(range(len(dataset)))
        test_indices_set = all_indices - train_indices_set
        train_indices_list = sorted(list(train_indices_set))
        test_indices_list = sorted(list(test_indices_set))
        train_subset = Subset(dataset, train_indices_list)
        test_subset = Subset(dataset, test_indices_list)
        return train_subset, test_subset

    # Prepare data list for BeamImageDataset
    # Each item in `data_list_for_dataset` represents ONE sample with its image and R-matrix parameters.
    data_list_for_dataset = []
    for i in range(len(to_use_indices)): # Iterate through the already-filtered data
        data_list_for_dataset.append({
            'image': processed_tensors[i], # This is a single 2D image tensor for this sample
            'r11': [filtered_r11[i]],       # A single r11 value for this sample
            'r12': [filtered_r12[i]],
            'r33': [filtered_r33[i]],
            'r34': [filtered_r34[i]],
            'quad_k': quad_torch_k_for_dataset[i] # A single quad_k value for this sample
        })

    # Create the full dataset from the prepared list
    full_filtered_dataset = BeamImageDataset(data_list_for_dataset)
    print(f"Total samples in filtered dataset: {len(full_filtered_dataset)}")

    # Define train and test indices for the split
    # "alternate image used for training and rest for testing"
    # This means: train = 0, 2, 4, ... ; test = 1, 3, 5, ...
    all_filtered_indices = list(range(len(full_filtered_dataset)))
    train_indices = [i for i in all_filtered_indices if i % 2 == 0]
    test_indices = [i for i in all_filtered_indices if i % 2 != 0]

    print(f"Train indices: {train_indices}")
    print(f"Test indices: {test_indices}")

    # Split the dataset
    train_ds, test_ds = split_dataset(full_filtered_dataset, train_indices)

    # Create DataLoaders for training and testing
    # Using the custom collate_fn for proper batching of dicts
    train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=False, collate_fn=beam_collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=5, shuffle=False, collate_fn=beam_collate_fn)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize ScreenImageGenerator
    # The r11, r12, r33, r34, quad_k provided here are the *initial* values used
    # if generate_screen_image (without params) is called.
    # For generate_screen_image_with_params (which we will use), the r-matrices
    # are passed per batch. So, these can be set to the *full set of filtered* values.
    screen_gen = ScreenImageGenerator(
        r11=filtered_r11, r12=filtered_r12, r33=filtered_r33, r34=filtered_r34, quad_k=filtered_quad_k,
        x_pixels=processed_tensors.shape[2], y_pixels=processed_tensors.shape[1], # Use actual image shape
        x_pixel_size=x_pixel_size_tensor.item(), y_pixel_size=y_pixel_size_tensor.item(), particle_batch_size= 500,
        device=device
    )

    # Initialize ParticleTransformer
    transformer = ParticleTransformer(
        input_dim=4,      # Changed from 6 to 4
        model_dim=16,
        n_heads=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=.1,
        output_dim=4,     # Changed from 6 to 4
        output_scale=5e-5
    )

    # Initialize NNParticleBeamGenerator
    beam_gen = NNParticleBeamGenerator(
        n_particles=40000,
        energy=p0c,  # Beam energy from HDF5
        transformer=transformer,
    )

    # Train the model with contamination loss
    trained_model = train_beam_generator_with_dataloader(
        beam_generator=beam_gen,
        screen_generator=screen_gen,
        train_dataloader=train_dataloader,
        n_epochs=11,
        lr=1e-2,
        # image_loss_weight=1.0,
        # containment_loss_weight=1e5,
        foreground_weight=10,
        background_weight=1,
        gaussian_blur_sigma=3,
        # total_particles=40000,
        device=device
    )
    # Get the reconstructed beam from the trained generator
    reconstructed_beam = trained_model.beam_generator()

    print("\nCalculating Emittance from Reconstructed Beam:")

    x_coords = reconstructed_beam.x.clone().detach()
    px_coords = reconstructed_beam.px.clone().detach()
    y_coords = reconstructed_beam.y.clone().detach()
    py_coords = reconstructed_beam.py.clone().detach()

    x_squared = torch.mean(x_coords**2)
    px_squared = torch.mean(px_coords**2)
    x_px_cross = torch.mean(x_coords * px_coords)

    y_squared = torch.mean(y_coords**2)
    py_squared = torch.mean(py_coords**2)
    y_py_cross = torch.mean(y_coords * py_coords)

    print(f"<x^2> = {x_squared.item():.4e}")
    print(f"<px^2> = {px_squared.item():.4e}")
    print(f"<xpx> = {x_px_cross.item():.4e}")
    print(f"<y^2> = {y_squared.item():.4e}")
    print(f"<py^2> = {py_squared.item():.4e}")
    print(f"<ypy> = {y_py_cross.item():.4e}")

    # Emittance calculation
    emittance_x = torch.sqrt(torch.abs(x_squared * px_squared - x_px_cross**2)) # Use abs for numerical stability
    alpha_x = -x_px_cross / emittance_x
    beta_x = x_squared / emittance_x
    gamma_x = px_squared / emittance_x # Add gamma_x for completeness

    print(f'\nReconstructed Emittance x: {emittance_x.item():.4e} m.rad')
    print(f'Reconstructed Alpha x: {alpha_x.item():.4f}')
    print(f'Reconstructed Beta x: {beta_x.item():.4f} m')
    # print(f'Reconstructed Gamma x: {gamma_x.item():.4f} rad/m')

    emittance_y = torch.sqrt(torch.abs(y_squared * py_squared - y_py_cross**2)) # Use abs for numerical stability
    alpha_y = -y_py_cross / emittance_y
    beta_y = y_squared / emittance_y
    gamma_y = py_squared / emittance_y # Add gamma_y for completeness

    print(f'\nReconstructed Emittance y: {emittance_y.item():.4e} m.rad')
    print(f'Reconstructed Alpha y: {alpha_y.item():.4f}')
    print(f'Reconstructed Beta y: {beta_y.item():.4f} m')
    # print(f'Reconstructed Gamma y: {gamma_y.item():.4f} rad/m')


    print("\nOriginal (Target) Values for Comparison:")

    print(f'Target Emittance x: {torch.tensor(x_emittance).item():.4e} m.rad')
    print(f'Target Alpha x: {torch.tensor(x_alpha).item():.4f}')
    print(f'Target Beta x: {torch.tensor(x_beta).item():.4f} m')

    print(f'\nTarget Emittance y: {torch.tensor(y_emittance).item():.4e} m.rad')
    print(f'Target Alpha y: {torch.tensor(y_alpha).item():.4f}')
    print(f'Target Beta y: {torch.tensor(y_beta).item():.4f} m')

    emittance_results = {
        'emittance_x': emittance_x.item(),
        'alpha_x': alpha_x.item(),
        'beta_x': beta_x.item(),
        'emittance_y': emittance_y.item(),
        'alpha_y': alpha_y.item(),
        'beta_y': beta_y.item(),
        'target_emittance_x': torch.tensor(x_emittance).item(),
        'target_emittance_y': torch.tensor(y_emittance).item(),
        'target_alpha_x': torch.tensor(x_alpha).item(),
        'target_alpha_y': torch.tensor(y_alpha).item(),
        'target_beta_x': torch.tensor(x_beta).item(),
        'target_beta_y': torch.tensor(y_beta).item()
    }

    # Save results to logs
    test_dataloader = DataLoader(train_ds, batch_size=11, shuffle=False, collate_fn=beam_collate_fn)
    test_batch = next(iter(test_dataloader))

    # Move test batch data to device
    target_images_test = test_batch['images'].to(device)
    r11_test = test_batch['r11'].to(device)
    r12_test = test_batch['r12'].to(device)
    r33_test = test_batch['r33'].to(device)
    r34_test = test_batch['r34'].to(device)
    quad_k_test = test_batch['quad_k'].to(device) # Quad K for plotting

    # Generate predicted images for the test set
    # It's crucial to put the model in evaluation mode and use torch.no_grad() for inference
    trained_model.eval()
    with torch.no_grad():
        predicted_images_test = trained_model(
            r11=r11_test, r12=r12_test, r33=r33_test, r34=r34_test, quad_k=quad_k_test
        )

    save_results_to_logs(reconstructed_beam=reconstructed_beam, 
                         target_images_test=target_images_test, 
                         predicted_images_test=predicted_images_test[:,0], 
                         quad_k_test=quad_k_test,
                         emittance_results=emittance_results,
                         logger_log_dir=f'{log_dir}/version_{version}')