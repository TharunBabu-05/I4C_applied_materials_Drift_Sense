#!/usr/bin/env python3
"""
Drift-Sense: Localization / Inference Script  (v2.5 -- Multi-Scale NCC Pyramid)
================================================================================

Given a reference image (1000x1000, 100x magnification) and a search image
(1000x1000, 10x magnification), finds the center (x, y) of the region in
the search image where the reference pattern appears (shrunk 10x).

Algorithm: Multi-Scale NCC Pyramid
  1. Preprocess (histogram equalization + light Gaussian denoise):
     - Reference: sigma=0.5  (light -- preserves fine 100x detail)
     - Search:    sigma=0.8  (slightly heavier -- attenuates 10x shot noise)
  2. Build a 3-level image pyramid:
     - Level 0 (Coarse):  50px template vs 500px search  (20x total factor)
     - Level 1 (Nominal): 100px template vs 1000px search (10x factor)
     - Level 2 (Fine):    200px template vs 400px window  (5x factor)
  3. Level 0: Fast NCC -> top-20 coarse candidates
  4. Level 1: Full NCC on search; fuse L0+L1 scores (35%+65%)
  5. Level 2: Sub-cell refinement in tight upscaled window
  6. Center-bias disambiguation for remaining tied peaks

Why this approach:
  - Simple Gaussian preprocessing preserves structural edges NCC depends on
  - Multi-scale handling avoids cell-level periodic aliasing at coarse levels
  - NCC (TM_CCOEFF_NORMED) is illumination-invariant (handles gain/offset drift)
  - No training data or GPU required -- works on unseen test data
  - Center-bias disambiguation follows physical drift distribution reasoning

References:
  [1] Lewis, "Fast Normalized Cross-Correlation," Vision Interface, 1995.
  [2] Adelson et al., "Pyramid methods in image processing," RCA Engineer, 1984.
  [3] Foroosh et al., "Extension of Phase Correlation to Subpixel Registration,"
      IEEE TIP 11(3), 2002.

Usage:
  python inference.py --reference path/to/reference.png --search path/to/search.png

Output:
  Prints the predicted center (x, y) of the reference pattern in the search image.

Author: Drift-Sense Team
"""

import argparse
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


# =============================================================================
# Preprocessing
# =============================================================================

def load_grayscale(path):
    """Load an image as grayscale float64 array."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)


def histogram_equalize(image):
    """
    Apply contrast-limited histogram equalization.
    Normalizes intensity distribution to reduce contrast differences
    between reference and search images captured at different magnifications.
    """
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
    return ndimage.gaussian_filter(image, sigma=sigma)


def median_denoise(image, size=3):
    """
    Apply median filter for defect-robust denoising.

    Median filtering is highly effective against salt-and-pepper noise caused
    by bright particle contamination and dark missing-contact defects. Unlike
    Gaussian filtering, it preserves structural edges (cell walls) while
    eliminating isolated bright/dark defect pixels.

    Reference: [1] Goldstein et al., 2018 -- SEM imaging noise types.
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    filtered = cv2.medianBlur(img_uint8, size)
    return filtered.astype(np.float64)


def bilateral_denoise(image, d=7, sigma_color=25, sigma_space=7):
    """
    Apply bilateral filter for edge-preserving denoising.

    The bilateral filter smooths homogeneous regions (inside capacitor bodies
    and along uniform wall segments) while preserving sharp intensity
    transitions at cell boundaries. This makes template matching more robust
    to SEM noise while keeping structural features crisp.

    Reference: Tomasi & Manduchi, "Bilateral Filtering for Gray and Color
    Images," ICCV 1998.
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    return filtered.astype(np.float64)


def preprocess(image, denoise_sigma=0.8):
    """Full preprocessing pipeline: histogram equalization + gentle Gaussian denoising."""
    img = histogram_equalize(image)
    img = light_denoise(img, sigma=denoise_sigma)
    return img


def preprocess_robust(image, median_size=3, bilateral_d=7, sigma_color=25,
                      sigma_space=7, gauss_sigma=1.2):
    """
    Defect-robust preprocessing pipeline for the search image.

    Pipeline order (tuned for defective DRAM SEM images):
      1. Histogram equalization -- normalises gain/offset variation
      2. Median filter (size=3) -- removes isolated defect pixels
         (bright particles, dark missing contacts)
      3. Bilateral filter -- edge-preserving smoothing of SEM shot noise
      4. Gaussian denoise -- final smoothing pass

    Heavier than the standard preprocess() to improve robustness on search
    images which have lower magnification (more noise per pixel) and may
    contain manufacturing defects.
    """
    img = histogram_equalize(image)
    img = median_denoise(img, size=median_size)
    img = bilateral_denoise(img, d=bilateral_d,
                            sigma_color=sigma_color, sigma_space=sigma_space)
    img = light_denoise(img, sigma=gauss_sigma)
    return img


# =============================================================================
# Image Pyramid Utilities
# =============================================================================

def resize_image(image, new_size):
    """
    Resize an image using high-quality Lanczos resampling.

    Parameters
    ----------
    image : np.ndarray
    new_size : tuple (width, height)

    Returns
    -------
    np.ndarray
    """
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil, dtype=np.float64)


def build_pyramid_level(reference, search, ref_target_size, search_target_size=None):
    """
    Build one level of the scale pyramid.

    Parameters
    ----------
    reference : np.ndarray
        Full 1000x1000 reference image.
    search : np.ndarray
        Full 1000x1000 search image.
    ref_target_size : int
        Target side length for the reference template (square).
    search_target_size : int, optional
        Target side length for the search image. If None, use search as-is.

    Returns
    -------
    template : np.ndarray
    search_resized : np.ndarray
    scale_factor : float
        Factor by which search was rescaled (for coordinate back-projection).
    """
    rh, rw = reference.shape
    template = resize_image(reference, (ref_target_size, ref_target_size))

    if search_target_size is not None and search_target_size != search.shape[1]:
        search_resized = resize_image(search, (search_target_size, search_target_size))
        scale_factor = search.shape[1] / search_target_size
    else:
        search_resized = search
        scale_factor = 1.0

    return template, search_resized, scale_factor


# =============================================================================
# NCC Template Matching
# =============================================================================

def ncc_search(search_image, template, top_k=20, min_score=-1.0):
    """
    Perform Fast Normalized Cross-Correlation using OpenCV.

    Uses TM_CCOEFF_NORMED which is exact normalized cross-correlation,
    illumination-invariant (handles gain/offset between images).

    Reference: [1] Lewis, 1995.

    Parameters
    ----------
    search_image : np.ndarray
    template : np.ndarray
    top_k : int
    min_score : float
        Minimum NCC score to accept as a candidate.

    Returns
    -------
    candidates : list of (y, x, score)
        In search_image coordinates (center of matched region).
    """
    search_f32 = search_image.astype(np.float32)
    template_f32 = template.astype(np.float32)

    res = cv2.matchTemplate(search_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape
    sh, sw = search_image.shape

    candidates = []
    res_copy = res.copy()

    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_copy)
        if max_val < min_score:
            break

        x, y = max_loc
        cy = y + th // 2
        cx = x + tw // 2
        candidates.append((cy, cx, float(max_val)))

        # Non-maximum suppression: mask out neighborhood
        y_lo = max(0, y - th // 2)
        y_hi = min(res_copy.shape[0], y + th // 2)
        x_lo = max(0, x - tw // 2)
        x_hi = min(res_copy.shape[1], x + tw // 2)
        res_copy[y_lo:y_hi, x_lo:x_hi] = -np.inf

    return candidates


# =============================================================================
# Disambiguation
# =============================================================================

def disambiguate(candidates, image_center=(500, 500), ncc_threshold=0.05):
    """
    Handle periodic ambiguity: if multiple candidates have very similar
    NCC scores (within threshold), pick the one closest to image center.

    Physical reasoning: mechanical drift in SEM stages follows a known
    distribution centered near the previous position, so the true landing
    site is statistically closer to the center than spurious periodic aliases.
    """
    if not candidates:
        return image_center
    if len(candidates) == 1:
        return (candidates[0][0], candidates[0][1])

    best_score = candidates[0][2]
    tied = [(y, x, s) for y, x, s in candidates if best_score - s < ncc_threshold]

    if len(tied) <= 1:
        return (candidates[0][0], candidates[0][1])

    cy, cx = image_center
    best = min(tied, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
    return (best[0], best[1])


# =============================================================================
# Multi-Scale NCC Pyramid (Main Pipeline)
# =============================================================================

def localize(reference_path, search_path, verbose=False):
    """
    Main inference function: find the reference pattern in the search image.

    Uses a 3-level coarse-to-fine NCC pyramid:

      Level 0 (Coarse, 2x downscale factor applied to search):
        - Template = reference downscaled to 50x50 (20x total scale)
        - Search   = search downscaled to 500x500
        - Purpose  : Fast exhaustive search, avoids periodic cell-level aliasing
        - Gets top-20 candidate regions

      Level 1 (Nominal, standard 10x scale):
        - Template = reference downscaled to 100x100
        - Search   = search at full 1000x1000
        - Purpose  : Standard-scale refinement, incorporates fine detail
        - Refines top-10 candidates from Level 0

      Level 2 (Fine, 5x total scale):
        - Template = reference downscaled to 200x200
        - Search   = search upscaled to 2000x2000 (2x)
        - Purpose  : Sub-cell precision on a tight window around best candidate
        - Final refinement step

    Each level votes on the best candidate, and a weighted fusion picks the winner.
    """
    start_time = time.time()

    # --- Load images ---
    if verbose:
        print("Loading images...")
    reference = load_grayscale(reference_path)
    search = load_grayscale(search_path)

    # --- Preprocess (v2 proven pipeline) ---
    # Simple histogram equalization + light Gaussian denoise.
    # The median/bilateral pipeline in v3 over-smoothed structural edges
    # that NCC depends on, causing a 24-point accuracy regression.
    if verbose:
        print("Preprocessing...")
    ref_proc = preprocess(reference, denoise_sigma=0.5)
    search_proc = preprocess(search, denoise_sigma=0.8)

    # =========================================================================
    # LEVEL 0: Coarse search at 2x downscaled search (20x total template scale)
    # =========================================================================
    if verbose:
        print("Level 0: Coarse NCC search (500x500 search, 50x50 template)...")

    template_l0, search_l0, scale_l0 = build_pyramid_level(
        ref_proc, search_proc,
        ref_target_size=50,
        search_target_size=500
    )
    # scale_l0 = 1000/500 = 2.0

    candidates_l0 = ncc_search(search_l0, template_l0, top_k=20, min_score=-0.5)

    if verbose:
        print(f"  Found {len(candidates_l0)} coarse candidates")

    # Scale candidates back to full 1000x1000 search coordinates
    candidates_l0_full = [
        (int(round(y * scale_l0)), int(round(x * scale_l0)), s)
        for y, x, s in candidates_l0
    ]

    # =========================================================================
    # LEVEL 1: Nominal-scale NCC at 100x100 template, full 1000x1000 search
    # =========================================================================
    if verbose:
        print("Level 1: Nominal NCC refinement (1000x1000 search, 100x100 template)...")

    template_l1, search_l1, scale_l1 = build_pyramid_level(
        ref_proc, search_proc,
        ref_target_size=100,
        search_target_size=None   # keep search at full size
    )
    # scale_l1 = 1.0

    # Run full NCC at this level; fuse with Level-0 candidates
    all_l1 = ncc_search(search_l1, template_l1, top_k=30, min_score=-0.5)

    # Find the Level-1 score for each Level-0 candidate by proximity
    def best_l1_near(cy, cx, radius=60):
        """Find the best Level-1 candidate within radius pixels."""
        best = None
        best_score = -np.inf
        for y, x, s in all_l1:
            if abs(y - cy) <= radius and abs(x - cx) <= radius and s > best_score:
                best_score = s
                best = (y, x, s)
        return best

    fused_candidates = []
    for cy0, cx0, s0 in candidates_l0_full:
        c1 = best_l1_near(cy0, cx0, radius=70)
        if c1 is not None:
            y1, x1, s1 = c1
            # Fused score: weight Level-1 (more detailed) higher
            fused_score = 0.35 * s0 + 0.65 * s1
            fused_candidates.append((y1, x1, fused_score))
        else:
            # No Level-1 match nearby: use Level-0 position with lower weight
            fused_candidates.append((cy0, cx0, s0 * 0.5))

    # Sort by fused score
    fused_candidates.sort(key=lambda c: -c[2])

    if verbose:
        print(f"  Top-5 fused candidates:")
        for i, (y, x, s) in enumerate(fused_candidates[:5]):
            print(f"    #{i+1}: ({x}, {y}) score={s:.4f}")

    # =========================================================================
    # LEVEL 2: Fine refinement in a tight window around the best candidate
    # =========================================================================
    best_y, best_x, best_score = fused_candidates[0] if fused_candidates else (500, 500, 0)

    if verbose:
        print(f"Level 2: Fine refinement around ({best_x}, {best_y})...")

    # Crop a 200x200 window from the full search around the best candidate
    # and upscale 2x to get 400x400 for fine matching
    win_half = 100
    wy0 = max(0, best_y - win_half)
    wy1 = min(search_proc.shape[0], best_y + win_half)
    wx0 = max(0, best_x - win_half)
    wx1 = min(search_proc.shape[1], best_x + win_half)

    search_window = search_proc[wy0:wy1, wx0:wx1]

    if search_window.shape[0] > 20 and search_window.shape[1] > 20:
        # Upscale window 2x
        fine_scale = 2
        win_h, win_w = search_window.shape
        search_fine = resize_image(search_window,
                                   (win_w * fine_scale, win_h * fine_scale))

        # Template at 200x200 (5x total scale from 1000x1000 reference)
        template_l2 = resize_image(ref_proc, (200, 200))

        if (search_fine.shape[0] >= template_l2.shape[0] and
                search_fine.shape[1] >= template_l2.shape[1]):
            fine_cands = ncc_search(search_fine, template_l2, top_k=5, min_score=-0.5)
            if fine_cands:
                # Best fine candidate: convert from fine-scale window coords back
                fy, fx, fs = fine_cands[0]
                # From fine-scale to original window coords: divide by fine_scale
                fy_win = fy / fine_scale
                fx_win = fx / fine_scale
                # From window coords to full image coords
                fy_full = int(round(wy0 + fy_win))
                fx_full = int(round(wx0 + fx_win))
                if verbose:
                    print(f"  Fine result: ({fx_full}, {fy_full}) score={fs:.4f}")
                # If fine score is good, override
                if fs > best_score - 0.1:
                    best_y, best_x = fy_full, fx_full

    # =========================================================================
    # Disambiguation: handle any remaining tied candidates
    # =========================================================================
    if verbose:
        print("Disambiguating...")

    image_center = (search_proc.shape[0] // 2, search_proc.shape[1] // 2)
    result_y, result_x = disambiguate(
        fused_candidates[:10],
        image_center=image_center,
        ncc_threshold=0.03
    )

    # Use Level-2 refinement result if it doesn't deviate too far from Level-1
    if abs(best_y - result_y) <= 15 and abs(best_x - result_x) <= 15:
        result_y, result_x = best_y, best_x

    # Clamp to valid range
    result_x = max(0, min(search_proc.shape[1] - 1, int(round(result_x))))
    result_y = max(0, min(search_proc.shape[0] - 1, int(round(result_y))))

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nResult: ({result_x}, {result_y})")
        print(f"Time: {elapsed:.3f}s")

    return (result_x, result_y)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Multi-Scale NCC Localization (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Finds the center (x, y) of the reference pattern within the search image.

Examples:
  python inference.py --reference ref.png --search search.png
  python inference.py --reference ref.png --search search.png --verbose
        """
    )
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to the 1000x1000 reference image")
    parser.add_argument("--search", type=str, required=True,
                        help="Path to the 1000x1000 search image")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed debug output")

    args = parser.parse_args()

    import os
    if not os.path.isfile(args.reference):
        print(f"ERROR: Reference image not found: {args.reference}")
        sys.exit(1)
    if not os.path.isfile(args.search):
        print(f"ERROR: Search image not found: {args.search}")
        sys.exit(1)

    x, y = localize(args.reference, args.search, verbose=args.verbose)
    print(f"({x}, {y})")


if __name__ == "__main__":
    main()
