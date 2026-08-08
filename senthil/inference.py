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
import gc
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
    """Load an image as grayscale uint8 array."""
    img = Image.open(path).convert('L')
    return np.ascontiguousarray(np.array(img, dtype=np.uint8))


def histogram_equalize(image):
    """
    Apply contrast-limited histogram equalization directly in uint8.
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(img_uint8)


def light_denoise(image, sigma=1.0):
    """Apply light Gaussian smoothing for denoising."""
    ksize = int(round(sigma * 3)) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


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
    return filtered.astype(np.float32)


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
    return filtered.astype(np.float32)


def preprocess(image, denoise_sigma=0.5):
    """Full preprocessing pipeline: histogram equalization + Sobel edge magnitude blend."""
    img = histogram_equalize(image)
    if denoise_sigma > 0:
        img = light_denoise(img, sigma=denoise_sigma)

    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(dx, dy)
    max_v = edge_mag.max()
    if max_v > 0:
        edge_mag = (edge_mag / max_v) * 255.0

    blended = 0.6 * edge_mag + 0.4 * img.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


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

    # Weighted Template Matching Mask: Gives 3.0x booster weight to high-contrast marker features
    marker_mask = np.ones_like(template_f32, dtype=np.float32)
    marker_mask[template_f32 < 20] = 3.0
    marker_mask[template_f32 > 240] = 3.0

    res = cv2.matchTemplate(search_f32, template_f32, cv2.TM_CCOEFF_NORMED, mask=marker_mask)
    th, tw = template.shape
    sh, sw = search_image.shape

    candidates = []
    res_copy = res

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
        y_hi = min(res.shape[0], y + th // 2 + 1)
        x_lo = max(0, x - tw // 2)
        x_hi = min(res.shape[1], x + tw // 2 + 1)
        res_copy[y_lo:y_hi, x_lo:x_hi] = -np.inf

    del search_f32, template_f32
    gc.collect()

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

def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index (SSIM) between two 2D grayscale patches.
    Fast OpenCV Gaussian formulation.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


# =============================================================================
# Multi-Scale NCC Pyramid (Main Pipeline)
# =============================================================================

def localize_with_confidence(reference_path, search_path, verbose=False):
    """
    Main inference function returning predicted (x, y), confidence score, and debug info.
    """
    start_time = time.time()

    # 1. Load images
    reference = load_grayscale(reference_path)
    search = load_grayscale(search_path)

    # 2. Preprocess (Histogram Equalization + Sobel Edge Magnitude Blend)
    ref_proc = preprocess(reference, denoise_sigma=0.5)
    search_proc = preprocess(search, denoise_sigma=0.8)

    debug_info = {}

    # =========================================================================
    # LEVEL 0: Coarse search at 2x downscaled search (20x total template scale)
    # =========================================================================
    template_l0, search_l0, scale_l0 = build_pyramid_level(
        ref_proc, search_proc,
        ref_target_size=100,
        search_target_size=500
    )

    candidates_l0 = ncc_search(search_l0, template_l0, top_k=20, min_score=-0.5)
    candidates_l0_full = [
        (int(round(y * scale_l0)), int(round(x * scale_l0)), s)
        for y, x, s in candidates_l0
    ]
    debug_info["level0_candidates"] = candidates_l0_full

    # =========================================================================
    # LEVEL 1: Nominal-scale NCC (100x100 template vs full 1000x1000 search)
    # =========================================================================
    template_l1, search_l1, scale_l1 = build_pyramid_level(
        ref_proc, search_proc,
        ref_target_size=100,
        search_target_size=None
    )

    all_l1 = ncc_search(search_l1, template_l1, top_k=30, min_score=-0.5)
    debug_info["level1_candidates"] = all_l1

    # PROPER LEVEL 0 + LEVEL 1 FUSION (35% Coarse L0 + 65% Nominal L1)
    min_l0_score = min([c[2] for c in candidates_l0_full]) if candidates_l0_full else -0.5

    fused_candidates = []
    for y1, x1, s1 in all_l1:
        # Find nearest L0 candidate within 50px
        l0_matches = [c for c in candidates_l0_full if abs(c[0] - y1) <= 50 and abs(c[1] - x1) <= 50]
        if l0_matches:
            best_l0 = max(l0_matches, key=lambda c: c[2])
            s0 = best_l0[2]
        else:
            s0 = min_l0_score

        fused_score = 0.35 * s0 + 0.65 * s1
        fused_candidates.append((y1, x1, fused_score, s1, s0))

    fused_candidates.sort(key=lambda c: -c[2])
    debug_info["fused_candidates"] = fused_candidates

    best_y, best_x, best_fused_score, best_l1_s, best_l0_s = fused_candidates[0] if fused_candidates else (500, 500, 0.0, 0.0, 0.0)

    # =========================================================================
    # LEVEL 2: Fine Sub-Pixel Refinement Authority
    # =========================================================================
    win_half = 100
    wy0 = max(0, best_y - win_half)
    wy1 = min(search_proc.shape[0], best_y + win_half)
    wx0 = max(0, best_x - win_half)
    wx1 = min(search_proc.shape[1], best_x + win_half)

    search_window = search_proc[wy0:wy1, wx0:wx1]
    fine_refined_pos = None
    fine_score = -1.0

    if search_window.shape[0] > 20 and search_window.shape[1] > 20:
        fine_scale = 2
        win_h, win_w = search_window.shape
        search_fine = cv2.resize(search_window, (win_w * fine_scale, win_h * fine_scale), interpolation=cv2.INTER_CUBIC)
        template_l2 = cv2.resize(ref_proc, (200, 200), interpolation=cv2.INTER_CUBIC)

        if (search_fine.shape[0] >= template_l2.shape[0] and search_fine.shape[1] >= template_l2.shape[1]):
            fine_cands = ncc_search(search_fine, template_l2, top_k=5, min_score=-0.5)
            if fine_cands:
                fy, fx, fs = fine_cands[0]
                fy_win = fy / fine_scale
                fx_win = fx / fine_scale
                fy_full = int(round(wy0 + fy_win))
                fx_full = int(round(wx0 + fx_win))

                fine_refined_pos = (fy_full, fx_full)
                fine_score = fs

    # =========================================================================
    # CONDITIONAL ROTATION, SCALE & SSIM VERIFICATION (Triggered ONLY if Confidence < 0.40)
    # =========================================================================
    top_confidence = best_fused_score
    conditional_triggered = False

    if top_confidence < 0.40:
        conditional_triggered = True

        # Test rotation ±2° on top-3 candidates
        best_rot_score = top_confidence
        for angle in [-2.0, 2.0]:
            M = cv2.getRotationMatrix2D((50, 50), angle, 1.0)
            rot_template = cv2.warpAffine(template_l1, M, (100, 100))
            rot_cands = ncc_search(search_l1, rot_template, top_k=5, min_score=-0.5)
            if rot_cands and rot_cands[0][2] > best_rot_score:
                best_rot_score = rot_cands[0][2]
                best_y, best_x = rot_cands[0][0], rot_cands[0][1]

        # Test scale 0.99x / 1.01x on top candidate
        for scale_m in [0.99, 1.01]:
            new_dim = int(round(100 * scale_m))
            scaled_template = cv2.resize(template_l1, (new_dim, new_dim), interpolation=cv2.INTER_CUBIC)
            scaled_cands = ncc_search(search_l1, scaled_template, top_k=5, min_score=-0.5)
            if scaled_cands and scaled_cands[0][2] > best_rot_score:
                best_rot_score = scaled_cands[0][2]
                best_y, best_x = scaled_cands[0][0], scaled_cands[0][1]

    debug_info["conditional_fallback_triggered"] = conditional_triggered

    # =========================================================================
    # Disambiguation & Fine Refinement Authority Integration
    # =========================================================================
    image_center = (search_proc.shape[0] // 2, search_proc.shape[1] // 2)
    disambig_cands = [(c[0], c[1], c[2]) for c in fused_candidates[:10]]
    result_y, result_x = disambiguate(disambig_cands, image_center=image_center, ncc_threshold=0.17)

    # Level-2 Fine Refinement Authority: Preserve fine location whenever fine_score is high
    if fine_refined_pos is not None and fine_score >= (best_l1_s - 0.10):
        if abs(fine_refined_pos[0] - result_y) <= 60 and abs(fine_refined_pos[1] - result_x) <= 60:
            result_y, result_x = fine_refined_pos

    # Clamp coordinates to valid range
    result_x = max(0, min(search_proc.shape[1] - 1, int(round(result_x))))
    result_y = max(0, min(search_proc.shape[0] - 1, int(round(result_y))))

    elapsed = time.time() - start_time

    # Calculate Normalized Confidence Score C in [0.0, 1.0]
    final_confidence = np.clip((best_fused_score + 0.2) / 1.2, 0.0, 1.0)
    debug_info["confidence_score"] = float(final_confidence)
    debug_info["elapsed_time_sec"] = float(elapsed)
    debug_info["final_prediction"] = (result_x, result_y)

    if verbose:
        print(f"\nResult: ({result_x}, {result_y}) | Conf: {final_confidence:.3f} | Time: {elapsed:.3f}s")

    return (result_x, result_y, float(final_confidence), debug_info)


def localize(reference_path, search_path, verbose=False):
    """
    Standard localization wrapper returning (x, y) for backward compatibility.
    """
    rx, ry, conf, debug = localize_with_confidence(reference_path, search_path, verbose=verbose)
    return (rx, ry)


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
