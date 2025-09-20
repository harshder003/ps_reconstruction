import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.nn.functional as F

def fft_cross_correlation(img1, img2):
    # Both img1, img2: (H, W) - guaranteed to be on same device now
    f1 = torch.fft.fft2(img1)
    f2 = torch.fft.fft2(img2)
    cross_corr = torch.fft.ifft2(f1 * f2.conj())
    cross_corr = torch.fft.fftshift(cross_corr.real)
    
    # Find peak
    max_idx = torch.nonzero(cross_corr == cross_corr.max(), as_tuple=False)[0]
    
    # FIX: Create tensor on the same device as cross_corr
    center = torch.tensor(cross_corr.shape, device=cross_corr.device, dtype=max_idx.dtype) // 2
    shift = center - max_idx
    
    return shift

def shift_and_pad(img, shift):
    # img: (H, W), shift: (2,)
    h, w = img.shape
    pad_h = int(abs(shift[0]))
    pad_w = int(abs(shift[1]))
    # Pad on both sides
    img_padded = F.pad(img, (pad_w, pad_w, pad_h, pad_h))
    # Roll to align
    img_shifted = torch.roll(img_padded, shifts=(int(shift[0]), int(shift[1])), dims=(0, 1))
    # Crop back to original size
    center_h = img_shifted.shape[0] // 2
    center_w = img_shifted.shape[1] // 2
    half_h = h // 2
    half_w = w // 2
    img_aligned = img_shifted[center_h-half_h:center_h+half_h+(h%2), center_w-half_w:center_w+half_w+(w%2)]
    return img_aligned

class WeightedLoss(nn.Module):
    def __init__(self, foreground_weight=10.0, background_weight=1.0, threshold=1e-6):
        """
        Weighted loss for background-subtracted images.
        
        Args:
            foreground_weight: Weight for non-zero (foreground) pixels
            background_weight: Weight for zero/near-zero (background) pixels  
            threshold: Threshold to distinguish foreground from background
        """
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.threshold = threshold
    
    def forward(self, predicted, target):
        # Create weight mask based on target image
        # Foreground pixels (non-zero) get higher weight
        foreground_mask = (torch.abs(target) > self.threshold).float()
        background_mask = 1.0 - foreground_mask
        
        weights = (foreground_mask * self.foreground_weight + 
                  background_mask * self.background_weight)
        
        # Compute weighted MSE loss
        squared_diff = (predicted - target) ** 2
        weighted_loss = weights * squared_diff
        
        return weighted_loss.mean()

class ImageSimilarityLoss(nn.Module):
    def __init__(self, mode="mse", foreground_weight=10.0, background_weight=1.0):
        super().__init__()
        if mode == "mse":
            self.loss = nn.MSELoss()
        elif mode == "l1":
            self.loss = nn.L1Loss()
        elif mode == "weighted":
            self.loss = WeightedLoss(foreground_weight, background_weight, threshold=2.1e-5)
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted: Tensor of shape (batch_size, height, width)
            target: Tensor of shape (batch_size, height, width)
        Returns:
            Scalar loss value.
        """
        # FIX: Ensure both tensors are on the same device as predicted
        if predicted.device != target.device:
            target = target.to(predicted.device)
            print(f"Moved target from {target.device} to {predicted.device}")
        
        aligned_predicted = []
        for pred_img, tgt_img in zip(predicted, target):
            # Both images are now guaranteed to be on the same device
            shift = fft_cross_correlation(pred_img, tgt_img)
            pred_aligned = shift_and_pad(pred_img, shift)
            aligned_predicted.append(pred_aligned)
        
        aligned_predicted = torch.stack(aligned_predicted, dim=0)
        return self.loss(aligned_predicted, target) * 1e6

def normalize_images(images):
    """
    Normalizes images so that the pixel intensities of each image add up to 1.

    Parameters
    ----------
    images: torch.Tensor or list
        Tensor or list containing images

    Returns
    -------
    normalized images (same type as input)
    """
    # If images is a list, convert to tensor
    if isinstance(images, list):
        # Convert each image to a tensor if not already
        images = torch.stack([torch.as_tensor(img, dtype=torch.float32) for img in images], dim=0)
    else:
        images = images.float()
    sums = images.sum(dim=(-1, -2), keepdim=True)
    return images / sums

def kl_div(target, pred):
    eps = 1e-10
    return target * torch.abs((target + eps).log() - (pred + eps).log())

def log_mse(target, pred):
    eps = 1e-10
    return mse_loss((target + eps).log(), (pred + eps).log())

def mae_loss(target, pred):
    return torch.mean(torch.abs(target - pred)) * 1e2

def mae_log_loss(target, pred):
    return torch.mean(torch.abs(torch.log(target + 1e-8) - torch.log(pred + 1e-8)))