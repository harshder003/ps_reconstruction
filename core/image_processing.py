import numpy as np
import torch
from typing import Union, Tuple, Optional, List
from scipy.optimize import curve_fit

def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

def find_bright_spot_center_gaussian(image: np.ndarray) -> tuple[int, int]:
    """Find bright spot center by fitting Gaussian in x and y dimensions separately."""
    if image.ndim == 3:
        # If image has channels, average over channels
        image = image.mean(axis=0)

    # Sum over y-axis to get x profile and over x-axis to get y profile
    x_profile = image.sum(axis=0)
    y_profile = image.sum(axis=1)

    x = np.arange(len(x_profile))
    y = np.arange(len(y_profile))

    # Initial guesses for Gaussian parameters: amplitude, mean, sigma
    mean_x = np.sum(x * x_profile) / np.sum(x_profile)
    sigma_x = np.sqrt(np.sum(x_profile * (x - mean_x)**2) / np.sum(x_profile))
    initial_guess_x = [x_profile.max(), mean_x, sigma_x]

    mean_y = np.sum(y * y_profile) / np.sum(y_profile)
    sigma_y = np.sqrt(np.sum(y_profile * (y - mean_y)**2) / np.sum(y_profile))
    initial_guess_y = [y_profile.max(), mean_y, sigma_y]

    try:
        popt_x, _ = curve_fit(gaussian, x, x_profile, p0=initial_guess_x)
        popt_y, _ = curve_fit(gaussian, y, y_profile, p0=initial_guess_y)
        x_center = int(np.round(popt_x[1]))  # Use mean parameter from fit
        y_center = int(np.round(popt_y[1]))
    except Exception:
        # Fallback to max if fit fails
        max_idx = np.unravel_index(np.argmax(image), image.shape)
        y_center, x_center = max_idx

    return y_center, x_center

def crop_images_to_bright_spot_centers(
    images: Union[np.ndarray, torch.Tensor],
    target_x: int,
    target_y: int,
    x_axis: Optional[np.ndarray] = None,
    y_axis: Optional[np.ndarray] = None
) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """
    Crops each image in the input batch to the specified size (target_x, target_y), centered on its own bright spot.
    
    Args:
        images: Batch of images as a NumPy array or PyTorch tensor with shape (N, H, W) or (N, C, H, W).
        target_x: Target width (number of columns) of the cropped images.
        target_y: Target height (number of rows) of the cropped images.
        x_axis: Optional 1D NumPy array representing the x-axis values (e.g., for plotting).
        y_axis: Optional 1D NumPy array representing the y-axis values (e.g., for plotting).

    Returns:
        - List of cropped images (same type as input).
        - List of adjusted x_axis arrays (if provided).
        - List of adjusted y_axis arrays (if provided).
    """
    is_torch = isinstance(images, torch.Tensor)
    if is_torch:
        device = images.device
        images_np = images.detach().cpu().numpy()
    else:
        images_np = images

    # Determine shape and channels
    if len(images_np.shape) == 3:
        # (N, H, W) grayscale images
        N, h, w = images_np.shape
        channels = 1
    elif len(images_np.shape) == 4:
        # (N, C, H, W) multi-channel images
        N, c, h, w = images_np.shape
        channels = c
    else:
        raise ValueError("Input images must be 3D or 4D arrays.")

    if target_x > w or target_y > h:
        raise ValueError("Target dimensions cannot exceed original image dimensions.")

    cropped_images = []
    cropped_x_axes = []
    cropped_y_axes = []

    for i in range(N):
        if channels == 1:
            img = images_np[i]
        else:
            img = images_np[i]

        # Find bright spot center for THIS specific image
        center_y, center_x = find_bright_spot_center_gaussian(img)

        # Compute cropping indices around bright spot center
        start_x = max(center_x - target_x // 2, 0)
        end_x = start_x + target_x
        if end_x > w:
            end_x = w
            start_x = end_x - target_x

        start_y = max(center_y - target_y // 2, 0)
        end_y = start_y + target_y
        if end_y > h:
            end_y = h
            start_y = end_y - target_y

        # Crop the image
        if channels == 1:
            cropped_img = img[start_y:end_y, start_x:end_x]
        else:
            cropped_img = img[:, start_y:end_y, start_x:end_x]

        # Normalize cropped image
        if is_torch:
            cropped_img = torch.from_numpy(cropped_img).to(device)
            sums = cropped_img.sum(dim=(-1, -2), keepdim=True)
            cropped_img = cropped_img / sums
        else:
            if channels == 1:
                total_sum = cropped_img.sum()
                if total_sum > 0:
                    cropped_img = cropped_img / total_sum
            else:
                total_sum = cropped_img.sum(axis=(-2, -1), keepdims=True)
                total_sum[total_sum == 0] = 1
                cropped_img = cropped_img / total_sum

        cropped_images.append(cropped_img)
        stacked_images = np.array(cropped_images)

        # Adjust x_axis and y_axis for THIS specific image
        if x_axis is not None:
            cropped_x_axes.append(x_axis[start_x:end_x])
        else:
            cropped_x_axes.append(None)

        if y_axis is not None:
            cropped_y_axes.append(y_axis[start_y:end_y])
        else:
            cropped_y_axes.append(None)

    # Clear cache and collect garbage to reduce memory usage
    if is_torch:
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    return stacked_images, cropped_x_axes, cropped_y_axes