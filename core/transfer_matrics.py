import torch

def drift_matrix(L: torch.Tensor) -> torch.Tensor:
    """
    Create transfer matrix for a drift space with length L using PyTorch tensors.

    Args:
        L (torch.Tensor): Drift length in meters

    Returns:
        torch.Tensor: 4x4 transfer matrix for the drift
    """
    R = torch.tensor([
        [1, L],
        [0, 1]
    ])
    return torch.block_diag(R, R)

def quad_matrix(k: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Create transfer matrix for a quadrupole with strength k and length L using PyTorch tensors.
    Treats x and y independently with no cross-coupling.

    Args:
        k (torch.Tensor): Quadrupole strength (positive for focusing in x, negative for defocusing)
        L (torch.Tensor): Quadrupole length in meters

    Returns:
        torch.Tensor: 4x4 transfer matrix for the quadrupole
    """
    k_scalar = k.item()

    # Calculate x-plane transfer matrix
    if k_scalar > 0:  # Focusing in x
        sqrt_k_L = torch.sqrt(k) * L
        Rx = torch.tensor([
            [torch.cos(sqrt_k_L), (1/torch.sqrt(k))*torch.sin(sqrt_k_L)],
            [-torch.sqrt(k)*torch.sin(sqrt_k_L), torch.cos(sqrt_k_L)]
        ])
    else:  # Defocusing in x
        k_abs = torch.abs(k)
        sqrt_k_abs_L = torch.sqrt(k_abs) * L
        Rx = torch.tensor([
            [torch.cosh(sqrt_k_abs_L), (1/torch.sqrt(k_abs))*torch.sinh(sqrt_k_abs_L)],
            [torch.sqrt(k_abs)*torch.sinh(sqrt_k_abs_L), torch.cosh(sqrt_k_abs_L)]
        ])

    # Calculate y-plane transfer matrix (opposite focusing behavior)
    if k_scalar > 0:  # Defocusing in y
        sqrt_k_L = torch.sqrt(k) * L
        Ry = torch.tensor([
            [torch.cosh(sqrt_k_L), (1/torch.sqrt(k))*torch.sinh(sqrt_k_L)],
            [torch.sqrt(k)*torch.sinh(sqrt_k_L), torch.cosh(sqrt_k_L)]
        ])
    else:  # Focusing in y
        k_abs = torch.abs(k)
        sqrt_k_abs_L = torch.sqrt(k_abs) * L
        Ry = torch.tensor([
            [torch.cos(sqrt_k_abs_L), (1/torch.sqrt(k_abs))*torch.sin(sqrt_k_abs_L)],
            [-torch.sqrt(k_abs)*torch.sin(sqrt_k_abs_L), torch.cos(sqrt_k_abs_L)]
        ])

    # Create a block diagonal matrix with no cross-coupling between x and y
    R = torch.zeros((4, 4))
    R[0:2, 0:2] = Rx  # x-plane
    R[2:4, 2:4] = Ry  # y-plane

    return R