#!/usr/bin/env python3
import argparse
from PIL import Image
import numpy as np

def reduce_vertical_whitespace(input_path: str,
                               output_path: str,
                               threshold: int = 250,
                               gap: int = 0):
    """
    Trims large white gaps between content rows in an image.

    Args:
        input_path:  Path to the source image.
        output_path: Path where the trimmed image will be saved.
        threshold:   Pixel intensity above which a channel is considered 'white'.
        gap:         Number of white pixels to leave between segments.
    """
    # Load image and convert to RGB array
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    # Build a mask of 'white' pixels (all channels >= threshold)
    white_mask = np.all(arr >= threshold, axis=2)
    # Identify rows that are entirely white
    white_rows = np.all(white_mask, axis=1)

    # Find contiguous non-white row segments
    segments = []
    in_seg = False
    for y in range(h):
        if not white_rows[y] and not in_seg:
            in_seg = True
            start = y
        elif white_rows[y] and in_seg:
            in_seg = False
            segments.append((start, y))
    if in_seg:
        segments.append((start, h))

    if not segments:
        print("No non-white content found. Saving original.")
        img.save(output_path)
        return

    # Calculate total height of new image
    total_height = sum(end - start for start, end in segments)
    total_height += gap * (len(segments) - 1)

    # Create a new blank (white) image
    new_img = Image.new("RGB", (w, total_height), (255, 255, 255))

    # Paste each segment in order
    y_offset = 0
    for (start, end) in segments:
        crop = img.crop((0, start, w, end))
        new_img.paste(crop, (0, y_offset))
        y_offset += (end - start) + gap

    # Save the result
    new_img.save(output_path)
    print(f"Saved trimmed image to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduce vertical white space in an image of plots/comparisons."
    )
    parser.add_argument("input_path",  help="Path to the input image")
    parser.add_argument("output_path", help="Path to save the trimmed image")
    parser.add_argument(
        "--threshold", "-t",
        type=int, default=250,
        help="White-channel threshold (0–255) to detect background rows"
    )
    parser.add_argument(
        "--gap", "-g",
        type=int, default=0,
        help="Pixels of gap to leave between trimmed segments"
    )
    args = parser.parse_args()

    reduce_vertical_whitespace(
        args.input_path,
        args.output_path,
        threshold=args.threshold,
        gap=args.gap
    )