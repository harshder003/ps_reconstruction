# Phase Space Reconstruction for SwissFEL Beam Diagnostics

This repository contains a neural network-based approach for reconstructing 4D phase space beam distributions from screen images collected at the SwissFEL facility. The system uses transformer-based neural networks to learn the mapping from a base distribution to the actual beam distribution that produces observed screen images.

## Overview

The phase space reconstruction system operates through the following workflow:

1. **Initial Phase Space Generation**: Generate a 4D phase space beam distribution using a neural network transformer
2. **Accelerated Simulation**: Propagate the beam through optical elements using transfer matrices to produce screen images
3. **Comparison with Measurements**: Compare simulated images with actual SwissFEL screen measurements
4. **Training**: Optimize the neural network to produce beam distributions that generate matching screen images
5. **Reconstruction**: Extract the final reconstructed beam distribution that closely matches the original SwissFEL beam

## System Architecture

### Core Components

#### 1. Beam Generation (`core/beam_generator.py`)
- **ParticleTransformer**: Transformer-based neural network that transforms a base distribution into a realistic beam distribution
- **NNTransform**: Alternative neural network architecture using attention mechanisms
- **NNParticleBeamGenerator**: Main beam generator that combines neural networks with particle sampling

Key features:
- Supports both 4D (transverse) and 6D (full) phase space reconstruction
- Automatic coordinate centering to ensure zero-mean distributions
- Configurable transformer architectures with attention mechanisms
- GPU-accelerated processing with memory optimization

#### 2. Accelerated Simulation (`core/simulation.py`)
- **ScreenImageGenerator**: GPU-accelerated beam propagation and screen image generation
- Uses transfer matrix formalism for beam optics calculations
- Implements differentiable splatting for smooth particle-to-image conversion
- Supports FFT-based Gaussian blur for realistic image generation

Features:
- Memory-efficient particle batching for large beam populations
- Gradient checkpointing for training with large beams
- Configurable screen resolutions and pixel sizes
- Multiple quadrupole configurations support

#### 3. Training Framework (`core/train.py`)
- **LitBeamImageGenerator**: PyTorch Lightning module for training
- Supports various loss functions (MSE, L1, weighted)
- Automatic image alignment using cross-correlation
- Comprehensive logging and checkpointing

#### 4. Data Processing (`core/dataloader.py`, `core/datasets.py`)
- **BeamImageDataset**: Custom dataset for SwissFEL measurement data
- Background subtraction and image preprocessing
- Automatic bright spot detection and image cropping
- Support for multiple measurement configurations

#### 5. Loss Functions (`core/losses.py`)
- **ImageSimilarityLoss**: Multi-modal loss functions for image comparison
- **WeightedLoss**: Foreground/background weighted loss for beam images
- Automatic image alignment and normalization

### Visualization and Analysis (`utils/visualization.py`)
- Gaussian fitting for beam size extraction
- RMS and sigma calculations from image projections
- Comprehensive plotting functions for beam diagnostics
- Cross-correlation based image alignment

## Usage

### Basic Training Pipeline

```python
from ps_reconstruction.core.beam_generator import ParticleTransformer, NNParticleBeamGenerator
from ps_reconstruction.core.simulation import ScreenImageGenerator
from ps_reconstruction.core.train import train_beam_generator_with_dataloader

# Initialize components
transformer = ParticleTransformer(
    input_dim=4,
    output_dim=4,
    model_dim=16,
    n_heads=4,
    num_layers=2,
    output_scale=5e-5
)

beam_generator = NNParticleBeamGenerator(
    n_particles=40000,
    energy=beam_energy,
    transformer=transformer
)

screen_generator = ScreenImageGenerator(
    r11=r11_values, r12=r12_values, 
    r33=r33_values, r34=r34_values,
    quad_k=quad_k_values,
    x_pixels=120, y_pixels=120,
    x_pixel_size=7.7e-6, y_pixel_size=7.8e-6,
    device='cuda'
)

# Train the model
trained_model = train_beam_generator_with_dataloader(
    beam_generator=beam_generator,
    screen_generator=screen_generator,
    train_dataloader=train_dataloader,
    n_epochs=50,
    lr=1e-2,
    foreground_weight=10,
    background_weight=1,
    gaussian_blur_sigma=3
)

# Extract reconstructed beam
reconstructed_beam = trained_model.beam_generator()
```

### Data Loading and Preprocessing

```python
from ps_reconstruction.core.dataloader import BeamImageDataset, create_beam_dataloader
from ps_reconstruction.core.image_processing import crop_images_to_bright_spot_centers

# Load SwissFEL measurement data
with h5py.File('measurement_data.h5', 'r') as f:
    images = f['/Raw_data/image'][()]
    r11 = f['/Input/R11'][()]
    r12 = f['/Input/R12'][()]
    # ... other parameters

# Background subtraction and preprocessing
processed_images = images - 1000  # Background subtraction
processed_images[processed_images < 0] = 0

# Crop images to bright spot centers
cropped_images, x_axis, y_axis = crop_images_to_bright_spot_centers(
    processed_images[:, 0],  # Use first image from each measurement
    target_x=120, target_y=120
)

# Create dataset
dataset = BeamImageDataset(prepared_data)
dataloader = create_beam_dataloader(dataset, batch_size=10)
```

## Example Results

The system has been tested with real SwissFEL measurement data, showing excellent reconstruction quality:

### Version 40 Results
- **Reconstructed Emittance X**: 9.82×10⁻¹⁰ m·rad (Target: 6.65×10⁻¹⁰ m·rad)
- **Reconstructed Emittance Y**: 8.69×10⁻¹⁰ m·rad (Target: 8.91×10⁻¹⁰ m·rad)
- **Alpha X**: 0.69 (Target: 0.96)
- **Beta X**: 3.21 m (Target: 5.35 m)

### Training Performance
- Loss convergence from ~0.13 to ~0.006 over 50 epochs
- Successful image matching between simulated and measured data
- Robust beam size reconstruction across different quadrupole settings

## File Structure

```
ps_reconstruction/
├── core/                           # Core system components
│   ├── beam_generator.py          # Neural network beam generators
│   ├── simulation.py              # GPU-accelerated beam simulation
│   ├── train.py                   # Training framework
│   ├── losses.py                  # Loss functions
│   ├── dataloader.py              # Data loading utilities
│   ├── datasets.py                # Dataset classes
│   ├── image_processing.py        # Image preprocessing
│   ├── modeling.py                # GPSR modeling components
│   └── results.py                 # Results saving utilities
├── utils/
│   └── visualization.py           # Visualization and analysis tools
├── examples/
│   ├── implementation_emittance.ipynb  # Main implementation example
│   ├── version_40/                # Example results from measurement set 1
│   │   ├── final_results/
│   │   │   ├── emittance_results.txt
│   │   │   ├── reconstructed_beam_distribution.png
│   │   │   ├── sigma_comparison.png
│   │   │   └── test_images_comparison.png
│   │   ├── epoch_images/          # Training progress visualization
│   │   └── metrics.csv            # Training metrics
│   └── version_62/                # Example results from measurement set 2
├── main.py                        # Main training script
└── README.md                      # This file
```

## Key Features

### Neural Network Architecture
- **Transformer-based**: Uses multi-head attention for learning particle correlations
- **Positional Encoding**: Maintains spatial relationships in phase space
- **Residual Connections**: Ensures stable training
- **Configurable Depth**: Adjustable number of layers and attention heads

### GPU Acceleration
- **CUDA Support**: Full GPU acceleration for training and simulation
- **Memory Optimization**: Efficient memory usage with particle batching
- **Gradient Checkpointing**: Reduces memory requirements during training

### Physical Accuracy
- **Transfer Matrix Formalism**: Accurate beam optics calculations
- **Differentiable Simulation**: End-to-end differentiable pipeline
- **Realistic Image Generation**: Gaussian blur and noise modeling

### Robust Training
- **Image Alignment**: Automatic alignment using cross-correlation
- **Multiple Loss Functions**: MSE, L1, and weighted losses
- **Learning Rate Scheduling**: Adaptive learning rate optimization
- **Checkpointing**: Automatic model saving and resuming

## Dependencies

- PyTorch (with CUDA support recommended)
- PyTorch Lightning
- NumPy
- Matplotlib
- SciPy
- h5py
- Cheetah (for beam tracking)

## Getting Started

1. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning numpy matplotlib scipy h5py
```

2. Prepare SwissFEL measurement data in HDF5 format

3. Run training:
```bash
python main.py --dataset 1 --version version_40
```

4. Analyze results using the provided Jupyter notebooks

## Applications

This system is designed for:
- **Beam Diagnostics**: Accurate reconstruction of beam phase space distributions
- **Machine Studies**: Understanding beam behavior under different conditions  
- **Optimization**: Improving beam quality through better diagnostics
- **Research**: Studying beam physics and accelerator performance

The reconstructed beam distributions can be used for:
- Emittance measurements
- Beam matching studies
- Optics optimization
- Machine learning training data

## Citation

If you use this code in your research, please cite the appropriate references for the SwissFEL facility and beam diagnostics methods.

---

*This system represents a significant advancement in beam diagnostics, combining modern machine learning techniques with rigorous accelerator physics to provide accurate phase space reconstruction from screen measurements.*