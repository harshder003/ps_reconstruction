import torch
from torch import Tensor, nn
from typing import Optional, Union, Dict, Tuple

# Import from GPSR modeling package
from gpsr.modeling import GPSRLattice, GPSRQuadScanLattice, GPSR6DLattice, GenericGPSRLattice
from cheetah import Screen


class GPSRBeamSimulator(nn.Module):
    """
    A BeamSimulator compatible with GPSR lattice models that generates screen images
    by simulating beam propagation through magnetic elements.

    Supports:
        - Quad scan lattices (GPSRQuadScanLattice)
        - 5D/6D beams (GPSR6DLattice)
        - Custom lattices via GenericGPSRLattice
        - Dynamic screen creation from input parameters
    """

    def __init__(
        self,
        quad_k: Optional[Tensor] = None,
        l_quad: float = 0.2,
        l_drift: float = 1.0,
        lattice_type: str = "quad_scan",  # 'quad_scan', '6d', 'generic'
        image_shape: Tuple[int, int] = (100, 100),
        x_pixel_size: Optional[float] = None,
        y_pixel_size: Optional[float] = None,
        processed_x_axis: Optional[Tensor] = None,
        processed_y_axis: Optional[Tensor] = None,
        screen_method: str = "kde",
        kde_bandwidth: Optional[float] = None,
        upstream_elements=None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the GPSRBeamSimulator.

        Args:
            quad_k: Batched quadrupole strengths to simulate.
            l_quad: Length of quadrupole magnet.
            l_drift: Drift length after quadrupole.
            lattice_type: Type of lattice ('quad_scan', '6d', 'generic').
            image_shape: Shape of the output screen image (H, W).
            x_pixel_size: Horizontal pixel size in meters.
            y_pixel_size: Vertical pixel size in meters.
            processed_x_axis: X-axis data from real images (for computing pixel size).
            processed_y_axis: Y-axis data from real images (for computing pixel size).
            screen_method: Method to generate screen images ('kde' or 'histogram').
            kde_bandwidth: Bandwidth for KDE screen generation.
            upstream_elements: List of accelerator elements before the quad/drift.
            device: Device to run simulation on.
            dtype: Data type for tensors.
        """
        super().__init__()
        self.register_buffer("quad_k", quad_k)

        # Handle screen resolution and pixel size
        if x_pixel_size is None or y_pixel_size is None:
            assert (
                processed_x_axis is not None and processed_y_axis is not None
            ), "Provide either pixel sizes or processed axes"

            x_pixel_size = abs((processed_x_axis[-1] - processed_x_axis[0]) / image_shape[0]) * 1e-4
            y_pixel_size = abs((processed_y_axis[-1] - processed_y_axis[0]) / image_shape[1]) * 1e-4

        self.screen = Screen(
            resolution=image_shape,
            pixel_size=torch.tensor((x_pixel_size, y_pixel_size), dtype=dtype),
            method=screen_method,
            kde_bandwidth=torch.tensor((x_pixel_size + y_pixel_size) / 2, dtype=dtype),
            is_active=True,
        )

        # Build lattice based on type
        if lattice_type == "quad_scan":
            self.lattice = GPSRQuadScanLattice(l_quad=l_quad, l_drift=l_drift, screen=self.screen)
        elif lattice_type == "6d":
            # Example values for 6D lattice; customize as needed
            l_tdc = 0.05
            f_tdc = 2856e6
            phi_tdc = 0.0
            l_bend = 0.5
            theta_on = 0.0
            l1 = l_quad + l_drift + l_tdc
            l2 = 0.3
            l3 = 0.4
            self.lattice = GPSR6DLattice(
                l_quad=l_quad,
                l_tdc=l_tdc,
                f_tdc=f_tdc,
                phi_tdc=phi_tdc,
                l_bend=l_bend,
                theta_on=theta_on,
                l1=l1,
                l2=l2,
                l3=l3,
                screen_1=self.screen,
                screen_2=self.screen,
                upstream_elements=upstream_elements,
            )
        elif lattice_type == "generic":
            # Assume user provides custom lattice setup externally
            raise ValueError(
                "For 'generic' lattice type, use set_lattice() explicitly."
            )
        else:
            raise ValueError(f"Unsupported lattice_type: {lattice_type}")

        self.lattice.to(device=device)

    def set_lattice(self, lattice):
        """Set a custom lattice manually."""
        self.lattice = lattice

    def forward(self, beam) -> Tensor:
        """
        Simulate beam propagation through the lattice and return screen images.

        Args:
            beam: ParticleBeam object to propagate.

        Returns:
            Tensor of screen images, shape: (num_settings, H, W)
        """
        if self.quad_k is not None:
            self.lattice.set_lattice_parameters(self.quad_k.unsqueeze(-1))

        readings = self.lattice.track_and_observe(beam)

        # Stack results into tensor
        images = torch.stack([r.squeeze() for r in readings])
        # images = readings[0]

        # Ensure correct dimensions
        if images.dim() == 2:
            images = images.unsqueeze(0)

        return images

    def set_quad_parameters(self, quad_k: Tensor) -> None:
        """Update quadrupole strengths dynamically."""
        self.register_buffer("quad_k", quad_k)