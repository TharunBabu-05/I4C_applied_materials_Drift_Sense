#!/usr/bin/env python3
"""
Generate individual visualizations for all pairs in a dataset
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

def load_grayscale(path):
    """Load an image as grayscale float64 array."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)

def histogram_equalize(image):
    """Apply histogram equalization."""
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    hist, bins = np.histogram(img_uint8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total = image.size
    cdf_normalized = (cdf - cdf_min) / (total - cdf_min) * 255.0
    cdf_normalized = np.clip(cdf_normalized, 0, 255)
    equalized = cdf_normalized[img_uint8]
    return equalized.astype(np.float64)

def light_denoise(image, sigma=1.0):
    """Apply light Gaussian smoothing for denoising."""
    from scipy import ndimage
    return ndimage.gaussian_filter(image, sigma=sigma)

def preprocess(image, denoise_sigma=0.8):
    """Full preprocessing pipeline."""
    img = histogram_equalize(image)
    img = light_denoise(img, sigma=denoise_sigma)
    return img

def resize_image(image, new_size):
    """Resize an image using high-quality Lanczos resampling."""
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil, dtype=np.float64)

def compute_error(predicted, ground_truth):
    """Compute Euclidean pixel error between predicted and ground truth."""
    px, py = predicted
    gx = ground_truth["center_x"]
    gy = ground_truth["center_y"]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

def create_visualization(reference_path, search_path, predicted, ground_truth,
                         error, pair_name, output_path):
    """
    Create a side-by-side visualization showing reference, search image
    with predicted and true locations marked.
    """
    ref_img = Image.open(reference_path).convert('RGB')
    search_img = Image.open(search_path).convert('RGB')

    # Create combined image
    margin = 20
    combined_w = ref_img.width + search_img.width + margin * 3
    combined_h = max(ref_img.height, search_img.height) + margin * 2 + 60
    combined = Image.new('RGB', (combined_w, combined_h), (30, 30, 30))

    # Place images
    combined.paste(ref_img, (margin, margin + 40))
    combined.paste(search_img, (ref_img.width + margin * 2, margin + 40))

    draw = ImageDraw.Draw(combined)

    # Title
    draw.text((margin, 10), f"{pair_name} | Error: {error:.1f} px",
              fill=(255, 255, 255))
    draw.text((margin, combined_h - 20),
              "Reference (100x)", fill=(200, 200, 200))
    draw.text((ref_img.width + margin * 2, combined_h - 20),
              "Search (10x)", fill=(200, 200, 200))

    # Mark ground truth (green cross) on search image
    gx = ground_truth["center_x"] + ref_img.width + margin * 2
    gy = ground_truth["center_y"] + margin + 40
    cross_size = 8
    draw.line([(gx - cross_size, gy), (gx + cross_size, gy)],
              fill=(0, 255, 0), width=2)
    draw.line([(gx, gy - cross_size), (gx, gy + cross_size)],
              fill=(0, 255, 0), width=2)

    # Mark predicted location (red circle) on search image
    px = predicted[0] + ref_img.width + margin * 2
    py = predicted[1] + margin + 40
    radius = 6
    draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)],
                 outline=(255, 0, 0), width=2)

    # Draw ground truth bounding box (green dashed)
    half_t = 50  # template is 100×100
    gt_x1 = ground_truth["center_x"] - half_t + ref_img.width + margin * 2
    gt_y1 = ground_truth["center_y"] - half_t + margin + 40
    gt_x2 = gt_x1 + 100
    gt_y2 = gt_y1 + 100
    draw.rectangle([(gt_x1, gt_y1), (gt_x2, gt_y2)],
                   outline=(0, 255, 0), width=1)

    # Legend
    legend_x = combined_w - 250
    legend_y = 10
    draw.rectangle([(legend_x, legend_y), (legend_x + 12, legend_y + 12)],
                   outline=(0, 255, 0), width=2)
    draw.text((legend_x + 18, legend_y), "Ground Truth", fill=(0, 255, 0))
    draw.ellipse([(legend_x + 2, legend_y + 20), (legend_x + 12, legend_y + 30)],
                   outline=(255, 0, 0), width=2)
    draw.text((legend_x + 18, legend_y + 18), "Predicted", fill=(255, 0, 0))

    combined.save(str(output_path))

def main():
    data_dir = Path(r"C:\Semester-7\I4C_hackathon\generated_ring_dataset")
    results_dir = data_dir / "results"
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Get all pair directories
    pair_dirs = sorted([d for d in data_dir.iterdir()
                        if d.is_dir() and d.name.startswith("pair_")])
    
    print(f"Found {len(pair_dirs)} pairs to visualize")
    
    # Import inference module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from inference import localize
    
    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        ref_path = pair_dir / "reference.png"
        search_path = pair_dir / "search.png"
        gt_path = pair_dir / "ground_truth.json"
        
        if not all(p.exists() for p in [ref_path, search_path, gt_path]):
            print(f"Skipping {pair_name} - missing files")
            continue
        
        # Load ground truth
        with open(gt_path) as f:
            ground_truth = json.load(f)
        
        # Run inference
        try:
            predicted = localize(str(ref_path), str(search_path))
            error = compute_error(predicted, ground_truth)
            
            # Create visualization
            output_path = viz_dir / f"{pair_name}_visualization.png"
            create_visualization(ref_path, search_path, predicted, ground_truth,
                                 error, pair_name, output_path)
            print(f"[OK] Generated visualization for {pair_name} (error: {error:.1f}px)")
            
        except Exception as e:
            print(f"[FAIL] Failed to process {pair_name}: {e}")
    
    print(f"\nAll visualizations saved to: {viz_dir}")

if __name__ == "__main__":
    main()