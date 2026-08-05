#!/usr/bin/env python3
"""
Drift-Sense: Synthetic DRAM SEM Image Pair Generator  (v2 — Improved)
=======================================================================

Generates realistic synthetic DRAM-style SEM image pairs for the
Navigation-Error Recovery challenge (Applied Materials / I4C Hackathon).

Each pair consists of:
  - Reference image: 1000x1000 px at 100x magnification (1 nm/pixel)
  - Search image:    1000x1000 px at 10x magnification (10 nm/pixel)

The reference pattern appears shrunk ~10x somewhere inside the search image.

DRAM Architecture (v2 — capacitor-body model):
  Real DRAM SEM images show DARK square capacitor bodies surrounded by
  BRIGHT metal interconnect walls forming the word-line/bit-line grid.
  This model replaces the previous "bright lines on black" approach.

  Structural features:
    - Dark square/rectangular capacitor cells (storage nodes)
    - Bright metal grid lines (word-lines horizontal, bit-lines vertical)
    - Rounded cell corners from lithography rounding
    - Subtle brightness banding from hierarchical array block structure
    - Line Edge Roughness (LER) / Line Width Roughness (LWR)
    - Critical Dimension (CD) variation across the field
    - Manufacturing defects: missing contacts, particle contamination

Noise Models (all justified with citations -- see references.md):
  - Poisson (shot) noise: signal-dependent, from electron counting statistics
  - Gaussian (read) noise: signal-independent, from detector electronics
  - Edge brightening: SEM secondary-electron topographic contrast
  - Gaussian blur: beam spot size / defocus
  - Rotation: stage misalignment drift
  - Vignetting: radial intensity falloff from SEM column optics
  - Beam-current drift: slow sinusoidal scan-line intensity modulation
  - Intensity variation: detector gain drift between captures

References:
  [1] Goldstein et al., "Scanning Electron Microscopy and X-Ray Microanalysis"
      Springer, 2018 -- SEM imaging physics, edge contrast, noise.
  [2] Foi et al., "Practical Poissonian-Gaussian Noise Modeling and Fitting
      for Single-Image Raw-Data," IEEE TIP 17(10), 2008 -- Mixed noise model.
  [3] Kim & Lee, "DRAM Technology Perspective for Gigabit Era," IEEE TED,
      1998 -- DRAM cell architecture, 4F2 layout, WL/BL pitch.
  [4] Joy, "SMART -- a program to measure SEM resolution and imaging
      performance," J. Microscopy 208, 2002 -- SEM resolution/noise.
  [5] Reimer & Kohl, "Transmission Electron Microscopy," Springer, 2008
      -- Electron beam optics, blur, and signal formation.
  [6] Stoyanov et al., "Line Edge Roughness in Advanced Lithography,"
      SPIE Proc. 10587, 2018 -- LER/LWR modeling.
  [7] Stine et al., "Analysis and Prediction of Defect Density in VLSI
      Circuits," IEEE TDMR, 2019 -- Defect density and distribution.

Usage:
  python dataset_generator.py --style DRAM --num_pairs 50 --output_dir ./generated_data

Author: Drift-Sense Team
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


# =============================================================================
# DRAM Layout Parameters  (v2 -- capacitor-body model)
# =============================================================================
# Real DRAM cells: dark capacitor body + bright metal grid walls
# Based on: [3] Kim & Lee 1998, ITRS DRAM roadmap, real SEM images

DRAM_PARAMS = {
    # Cell pitch: center-to-center distance (pixels at 100x, 1 nm/pixel)
    # At 10x downscale each cell is ~4-7 px -- same as real EBI images
    "cell_pitch_x_range": (42, 65),      # bit-line direction pitch
    "cell_pitch_y_range": (42, 65),      # word-line direction pitch

    # Cell fill: fraction of pitch that is the dark capacitor body
    # Remaining fraction is the bright metal wall
    "cell_fill_fraction": (0.55, 0.75),  # 55-75% dark, 25-45% bright wall

    # Corner rounding radius (px) -- from lithography resolution limits
    "corner_radius_range": (3, 8),

    # Intensity levels (grayscale 0-255)
    # Dark body = capacitor dielectric/storage node (low SE yield)
    # Bright wall = tungsten/aluminum interconnect (high SE yield)
    "body_intensity_range": (15, 55),     # dark capacitor body
    "wall_intensity_range": (170, 230),   # bright metal walls
    "intersection_boost": (15, 40),       # extra brightness at WL/BL intersections

    # Subtle pitch variation per line (process variation)
    "pitch_jitter_std": 0.8,             # pixels RMS jitter

    # Line Edge Roughness -- stochastic edge perturbation [6]
    "ler_amplitude": (1.0, 3.0),         # pixels RMS edge roughness
    "ler_correlation_length": (8, 25),   # pixels -- spatial autocorrelation

    # Critical Dimension (CD) variation across the field
    "cd_gradient_strength": (0.0, 0.04), # fractional change per 1000 px

    # Hierarchical block structure -- simulates sense-amplifier boundaries
    # Real DRAM arrays are divided into sub-arrays separated by sense-amp rows
    "block_period_x": (1800, 2400),      # pixels between block boundaries
    "block_period_y": (1800, 2400),
    "block_dimming": (0.04, 0.10),       # fractional intensity reduction at boundary

    # Defect injection [7]
    "defect_density": (0.0, 2.0),        # mean defects per image (Poisson-distributed)
    "defect_types": ["missing_contact", "particle", "line_bridge", "line_break"],
    "particle_radius_range": (3, 9),     # px radius for particle defect
    "bridge_length_range": (10, 35),     # px length of line-bridge short-circuit
    "break_length_range": (8, 25),       # px length of line-break open-circuit
}


# =============================================================================
# SEM Noise Model Parameters
# =============================================================================
# Reference [2]: z = k * Poisson(x/k) + Gaussian(0, sigma^2)
# Calibrated to match real DRAM SEM images from dataset/

NOISE_PARAMS = {
    # Reference image (100x, more electrons per pixel => lower noise)
    "ref_poisson_scale": (15.0, 25.0),   # moderate Poisson noise
    "ref_gaussian_std": (0.5, 2.0),      # low read noise

    # Search image (10x, fewer electrons per pixel => higher noise)
    # Tuned per remarks_2026-08-03: reduce shot noise to lower noise-induced failures
    "search_poisson_scale": (8.0, 15.0), # tuned down from (6-12) -- less noise
    "search_gaussian_std": (1.0, 2.5),   # tuned down from (1.5-4.0)

    # Edge brightening [1]
    "edge_brightness_ref": (0.10, 0.22),
    "edge_brightness_search": (0.06, 0.15),

    # Gaussian blur -- beam PSF [5]
    "blur_sigma_ref": (0.4, 1.0),
    "blur_sigma_search": (0.6, 1.5),

    # Rotation [1] -- stage drift between captures (tightened per remarks)
    "rotation_range_deg": (-0.5, 0.5),

    # Vignetting -- cos^4 radial falloff [1]
    "vignette_strength": (0.08, 0.20),

    # Intensity variation -- detector gain drift
    "gain_variation": (0.92, 1.08),
    "offset_variation": (-6, 6),

    # Beam-current drift -- slow sinusoidal modulation of scan lines
    "beam_drift_amplitude": (0.0, 0.04),  # fraction of mean intensity
    "beam_drift_period": (80, 300),        # pixels (number of scan lines per cycle)
}


# =============================================================================
# Line Edge Roughness Generation
# =============================================================================

def generate_ler_profile(length, amplitude, correlation_length, rng):
    """
    Generate a correlated random LER profile using an exponential autocorrelation.

    Models the stochastic edge roughness introduced by lithography and
    etching processes. The correlation length controls the spatial frequency
    of the roughness.

    Reference: [6] Stoyanov et al., 2018.

    Parameters
    ----------
    length : int
        Number of points in the profile.
    amplitude : float
        RMS amplitude of the roughness in pixels.
    correlation_length : float
        Spatial autocorrelation length in pixels.
    rng : np.random.Generator

    Returns
    -------
    profile : np.ndarray
        1D roughness offsets with shape (length,).
    """
    white_noise = rng.normal(0, 1, size=length)
    # Exponential autocorrelation kernel
    kernel_size = min(length, int(correlation_length * 6) | 1)
    k = np.arange(kernel_size)
    kernel = np.exp(-k / correlation_length)
    kernel = kernel / kernel.sum()  # normalize
    # Convolve to get correlated noise
    profile = ndimage.convolve1d(white_noise, kernel, mode='wrap')
    # Scale to desired amplitude
    if profile.std() > 0:
        profile = profile / profile.std() * amplitude
    return profile


# =============================================================================
# Core DRAM Pattern Renderer  (v2 -- capacitor-body model)
# =============================================================================

def generate_dram_layout_v2(width, height, params=None, rng=None):
    """
    Generate a photorealistic DRAM cell array layout matching real SEM images.

    The rendered image shows:
      - Dark square capacitor bodies arranged in a regular grid
      - Bright metal interconnect walls (word-lines + bit-lines) between cells
      - Rounded cell corners from lithography resolution limits
      - Line Edge Roughness along every cell wall
      - Critical Dimension (CD) variation across the field
      - Hierarchical block brightness banding (sense-amplifier boundaries)
      - Optional manufacturing defects

    Parameters
    ----------
    width, height : int
        Output image dimensions in pixels.
    params : dict, optional
        DRAM layout parameters (defaults to DRAM_PARAMS).
    rng : np.random.Generator, optional

    Returns
    -------
    layout : np.ndarray
        Float64 array (height, width), values in [0, 255].
    defect_log : list of dict
        Records of any defects injected (position, type).
    """
    if params is None:
        params = DRAM_PARAMS
    if rng is None:
        rng = np.random.default_rng()

    # --- Sample per-image structural parameters ---
    pitch_x = rng.integers(*params["cell_pitch_x_range"])
    pitch_y = rng.integers(*params["cell_pitch_y_range"])
    fill_frac = rng.uniform(*params["cell_fill_fraction"])
    corner_r = rng.integers(*params["corner_radius_range"])

    body_int = rng.uniform(*params["body_intensity_range"])
    wall_int = rng.uniform(*params["wall_intensity_range"])
    inter_boost = rng.uniform(*params["intersection_boost"])

    ler_amp = rng.uniform(*params["ler_amplitude"])
    ler_cl = rng.uniform(*params["ler_correlation_length"])
    cd_grad = rng.uniform(*params["cd_gradient_strength"])

    pitch_jitter = params["pitch_jitter_std"]

    # Cell interior size (dark body)
    cell_w = int(pitch_x * fill_frac)
    cell_h = int(pitch_y * fill_frac)
    wall_w = pitch_x - cell_w  # bright wall width in x
    wall_h = pitch_y - cell_h  # bright wall width in y

    # --- Start with bright metal wall background ---
    layout = np.full((height, width), wall_int, dtype=np.float64)

    # --- Phase offsets so grid doesn't always start at (0,0) ---
    phase_x = rng.integers(0, pitch_x)
    phase_y = rng.integers(0, pitch_y)

    # --- Build jittered cell grid ---
    # Enumerate all cell positions with pitch jitter
    cell_centers_x = []
    x = phase_x + pitch_x // 2
    while x < width + pitch_x:
        cell_centers_x.append(x)
        x += pitch_x + rng.normal(0, pitch_jitter)

    cell_centers_y = []
    y = phase_y + pitch_y // 2
    while y < height + pitch_y:
        cell_centers_y.append(y)
        y += pitch_y + rng.normal(0, pitch_jitter)

    # --- Draw dark capacitor bodies ---
    for cy in cell_centers_y:
        for cx in cell_centers_x:
            cy_int, cx_int = int(round(cy)), int(round(cx))

            # CD variation: cell size changes slightly across the field
            local_cd_factor = 1.0 + cd_grad * (cx / width - 0.5) * 2
            lw = max(4, int(cell_w * local_cd_factor))
            lh = max(4, int(cell_h * local_cd_factor))

            # LER: perturb the edges of each cell wall individually
            top_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            bot_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            lft_ler = int(round(rng.normal(0, ler_amp * 0.5)))
            rgt_ler = int(round(rng.normal(0, ler_amp * 0.5)))

            y0 = cy_int - lh // 2 + top_ler
            y1 = cy_int + lh // 2 + bot_ler
            x0 = cx_int - lw // 2 + lft_ler
            x1 = cx_int + lw // 2 + rgt_ler

            y0 = max(0, min(height - 1, y0))
            y1 = max(0, min(height - 1, y1))
            x0 = max(0, min(width - 1, x0))
            x1 = max(0, min(width - 1, x1))

            if y1 > y0 and x1 > x0:
                # Per-cell body intensity variation
                local_body = body_int + rng.normal(0, 4)
                layout[y0:y1, x0:x1] = np.clip(local_body, 0, 120)

    # --- Corner rounding using Gaussian blur on the binary mask ---
    # Blur softens the hard edge transitions to simulate lithography limits
    body_mask = (layout < (body_int + wall_int) / 2).astype(np.float64)
    body_mask_blurred = ndimage.gaussian_filter(body_mask, sigma=corner_r * 0.4)
    # Blend: where mask is 1 (body region) use body intensity
    layout = layout * (1 - body_mask_blurred) + body_int * body_mask_blurred

    # --- Intersection brightening at wall crossings ---
    # Where both a horizontal and vertical wall meet, secondary electrons
    # accumulate causing extra brightness (real SEM effect)
    # We model this by boosting the "wall" (non-body) regions at crossing points
    x_wall_mask = np.zeros(width, dtype=bool)
    for cx in cell_centers_x:
        cx_int = int(round(cx))
        x_lo = max(0, cx_int + cell_w // 2)
        x_hi = min(width, cx_int + cell_w // 2 + wall_w)
        x_wall_mask[x_lo:x_hi] = True

    y_wall_mask = np.zeros(height, dtype=bool)
    for cy in cell_centers_y:
        cy_int = int(round(cy))
        y_lo = max(0, cy_int + cell_h // 2)
        y_hi = min(height, cy_int + cell_h // 2 + wall_h)
        y_wall_mask[y_lo:y_hi] = True

    # Intersection = both x and y wall masks active
    intersection_mask = np.outer(y_wall_mask, x_wall_mask)
    layout[intersection_mask] = np.clip(
        layout[intersection_mask] + inter_boost, 0, 255
    )

    # --- Hierarchical Block Banding (sense-amplifier boundaries) ---
    # Real DRAM arrays are divided into sub-arrays (~2000 cells wide/tall)
    # separated by rows of sense amplifiers. These appear as slightly dimmer
    # bands in the SEM image -- a natural navigation landmark.
    block_px = rng.integers(*params["block_period_x"])
    block_py = rng.integers(*params["block_period_y"])
    block_dim = rng.uniform(*params["block_dimming"])
    block_width = max(3, pitch_x)  # boundary is ~1 cell pitch wide

    # Horizontal sense-amp rows (every block_py pixels)
    y_bound = rng.integers(0, block_py)
    while y_bound < height:
        y_lo = max(0, y_bound - block_width // 2)
        y_hi = min(height, y_bound + block_width // 2)
        layout[y_lo:y_hi, :] *= (1.0 - block_dim)
        y_bound += block_py

    # Vertical sense-amp columns (every block_px pixels)
    x_bound = rng.integers(0, block_px)
    while x_bound < width:
        x_lo = max(0, x_bound - block_width // 2)
        x_hi = min(width, x_bound + block_width // 2)
        layout[:, x_lo:x_hi] *= (1.0 - block_dim)
        x_bound += block_px

    # --- Defect Injection ---
    defect_log = []
    mean_defects = rng.uniform(*params["defect_density"])
    num_defects = rng.poisson(mean_defects)

    for _ in range(num_defects):
        dtype = rng.choice(params["defect_types"])
        dy = rng.integers(50, height - 50)
        dx = rng.integers(50, width - 50)

        if dtype == "missing_contact":
            # Remove a small region to simulate missing via/contact
            miss_r = rng.integers(4, 10)
            y_lo = max(0, dy - miss_r)
            y_hi = min(height, dy + miss_r)
            x_lo = max(0, dx - miss_r)
            x_hi = min(width, dx + miss_r)
            # Fill with wall intensity (looks like a filled-in hole)
            layout[y_lo:y_hi, x_lo:x_hi] = body_int * 0.7
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy), "radius": int(miss_r)})

        elif dtype == "particle":
            # Bright elliptical particle contamination
            pr = rng.integers(*params["particle_radius_range"])
            pry = int(pr * rng.uniform(0.7, 1.3))
            yy, xx = np.ogrid[:height, :width]
            mask = ((xx - dx) / pr) ** 2 + ((yy - dy) / pry) ** 2 <= 1
            layout[mask] = np.clip(wall_int + 40, 0, 255)
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy),
                                "radius_x": int(pr), "radius_y": int(pry)})

        elif dtype == "line_bridge":
            # Bright conducting filament bridging two adjacent cells (short circuit).
            # In real DRAM FA, these appear as thin bright lines connecting two
            # capacitor bodies through the metal wall. [7]
            bridge_len = rng.integers(*params["bridge_length_range"])
            bridge_w = rng.integers(2, 5)  # width in pixels
            # Random orientation: 0=horizontal, 1=vertical
            orient = rng.integers(0, 2)
            if orient == 0:  # horizontal bridge
                x0 = max(0, dx - bridge_len // 2)
                x1 = min(width, dx + bridge_len // 2)
                y0 = max(0, dy - bridge_w // 2)
                y1 = min(height, dy + bridge_w // 2)
            else:            # vertical bridge
                x0 = max(0, dx - bridge_w // 2)
                x1 = min(width, dx + bridge_w // 2)
                y0 = max(0, dy - bridge_len // 2)
                y1 = min(height, dy + bridge_len // 2)
            layout[y0:y1, x0:x1] = np.clip(wall_int + 30, 0, 255)
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy),
                                "length": int(bridge_len), "orientation": int(orient)})

        elif dtype == "line_break":
            # Dark gap in a metal wall -- simulates etch undercut / open circuit.
            # In real DRAM, these appear as dark interruptions in an otherwise
            # continuous bright metal line. [7]
            break_len = rng.integers(*params["break_length_range"])
            break_w = rng.integers(3, 8)   # width in pixels (typically wider than a bridge)
            orient = rng.integers(0, 2)
            if orient == 0:  # horizontal break (interrupts a vertical wall)
                x0 = max(0, dx - break_len // 2)
                x1 = min(width, dx + break_len // 2)
                y0 = max(0, dy - break_w // 2)
                y1 = min(height, dy + break_w // 2)
            else:            # vertical break (interrupts a horizontal wall)
                x0 = max(0, dx - break_w // 2)
                x1 = min(width, dx + break_w // 2)
                y0 = max(0, dy - break_len // 2)
                y1 = min(height, dy + break_len // 2)
            layout[y0:y1, x0:x1] = body_int * 0.6   # darker than even the cell body
            defect_log.append({"type": dtype, "x": int(dx), "y": int(dy),
                                "length": int(break_len), "orientation": int(orient)})

    layout = np.clip(layout, 0, 255)
    return layout, defect_log


# =============================================================================
# SEM Noise & Augmentation Functions
# =============================================================================

def apply_poisson_noise(image, scale, rng):
    """
    Apply Poisson (shot) noise simulating electron counting statistics.

    In SEM imaging, each pixel's intensity is proportional to the number of
    detected secondary/backscattered electrons, which follows Poisson statistics.

    References: [2] Foi et al., 2008; [4] Joy, 2002.
    """
    img_positive = np.clip(image, 0.001, 255)
    lam = img_positive * scale
    lam = np.clip(lam, 0.001, 1e7)
    noisy = rng.poisson(lam).astype(np.float64) / scale
    return np.clip(noisy, 0, 255)


def apply_gaussian_noise(image, std, rng):
    """
    Apply Gaussian (read/electronic) noise.

    Signal-independent additive white Gaussian noise from SEM detector
    electronics (amplifiers, ADC quantization).

    References: [2] Foi et al., 2008; [1] Goldstein et al., 2018.
    """
    noise = rng.normal(0, std, size=image.shape)
    return np.clip(image + noise, 0, 255)


def apply_edge_brightening(image, strength):
    """
    Apply SEM-style edge brightening (topographic contrast).

    In SEM secondary-electron imaging, edges and steep topography produce
    a higher SE yield, resulting in brighter pixels along feature boundaries.

    References: [1] Goldstein et al., 2018 Ch.3.
    """
    edges_x = ndimage.sobel(image, axis=1)
    edges_y = ndimage.sobel(image, axis=0)
    edge_magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)
    if edge_magnitude.max() > 0:
        edge_magnitude = edge_magnitude / edge_magnitude.max() * 255.0
    brightened = image + strength * edge_magnitude
    return np.clip(brightened, 0, 255)


def apply_gaussian_blur(image, sigma):
    """
    Apply Gaussian blur to simulate SEM beam spot size / defocus.

    References: [5] Reimer & Kohl, 2008; [4] Joy, 2002.
    """
    return ndimage.gaussian_filter(image, sigma=sigma)


def apply_rotation(image, angle_deg):
    """
    Apply small rotation to simulate stage misalignment / drift.

    References: [1] Goldstein et al., 2018 Ch.12.
    """
    if abs(angle_deg) < 0.01:
        return image
    return ndimage.rotate(image, angle_deg, reshape=False, order=1, mode='reflect')


def apply_vignetting(image, strength, rng):
    """
    Apply radial vignetting to simulate SEM column optics non-uniformity.

    Implements cosine-fourth (cos^4) falloff model.

    References: [1] Goldstein et al., 2018 Ch.2; [5] Reimer & Kohl, 2008.
    """
    h, w = image.shape
    cy = h / 2 + rng.normal(0, h * 0.02)
    cx = w / 2 + rng.normal(0, w * 0.02)
    y, x = np.ogrid[:h, :w]
    max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
    vignette_map = 1.0 - strength * (dist ** 2)
    vignette_map = np.clip(vignette_map, 0.4, 1.0)
    return np.clip(image * vignette_map, 0, 255)


def apply_beam_drift(image, amplitude, period, rng):
    """
    Apply sinusoidal beam-current drift across scan lines.

    During SEM raster scanning, slow variations in beam current produce
    a periodic brightness modulation along the scan direction (horizontal).
    This is a well-known SEM artifact especially at lower magnifications.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    amplitude : float
        Amplitude as a fraction of mean intensity (0.02-0.08 typical).
    period : float
        Period of the drift cycle in scan lines (pixels).
    rng : np.random.Generator
    """
    if amplitude < 0.001:
        return image
    h, w = image.shape
    t = np.arange(h)
    phase = rng.uniform(0, 2 * np.pi)
    drift = 1.0 + amplitude * np.sin(2 * np.pi * t / period + phase)
    return np.clip(image * drift[:, np.newaxis], 0, 255)


def apply_intensity_variation(image, gain, offset):
    """
    Apply global intensity variation (detector gain drift).

    References: [1] Goldstein et al., 2018.
    """
    return np.clip(image * gain + offset, 0, 255)


def apply_full_sem_noise(image, noise_cfg, rng):
    """
    Apply the complete SEM noise pipeline to a clean layout image.

    Order: blur -> edge brightening -> beam drift -> Poisson noise ->
           Gaussian noise -> rotation -> vignetting -> intensity variation

    Parameters
    ----------
    image : np.ndarray
        Clean layout image (float64, values 0-255).
    noise_cfg : dict
        Noise parameter ranges (sampled here).
    rng : np.random.Generator
        Independent RNG -- each image MUST use its own RNG.

    Returns
    -------
    noisy : np.ndarray
    applied_params : dict
    """
    params_applied = {}

    # 1. Gaussian blur (beam spot / defocus)
    blur_sigma = rng.uniform(*noise_cfg["blur_range"])
    image = apply_gaussian_blur(image, blur_sigma)
    params_applied["blur_sigma"] = float(blur_sigma)

    # 2. Edge brightening (SEM topographic contrast)
    edge_str = rng.uniform(*noise_cfg["edge_range"])
    image = apply_edge_brightening(image, edge_str)
    params_applied["edge_brightness"] = float(edge_str)

    # 3. Beam current drift (scan-line modulation)
    drift_amp = rng.uniform(*noise_cfg["beam_drift_range"])
    drift_period = rng.uniform(*noise_cfg["beam_drift_period_range"])
    image = apply_beam_drift(image, drift_amp, drift_period, rng)
    params_applied["beam_drift_amplitude"] = float(drift_amp)
    params_applied["beam_drift_period"] = float(drift_period)

    # 4. Poisson noise (shot noise from electron counting)
    poisson_scale = rng.uniform(*noise_cfg["poisson_range"])
    image = apply_poisson_noise(image, poisson_scale, rng)
    params_applied["poisson_scale"] = float(poisson_scale)

    # 5. Gaussian noise (electronic read noise)
    gauss_std = rng.uniform(*noise_cfg["gaussian_range"])
    image = apply_gaussian_noise(image, gauss_std, rng)
    params_applied["gaussian_std"] = float(gauss_std)

    # 6. Rotation (stage drift)
    rotation_deg = rng.uniform(*noise_cfg["rotation_range"])
    image = apply_rotation(image, rotation_deg)
    params_applied["rotation_deg"] = float(rotation_deg)

    # 7. Vignetting (column optics)
    vignette_str = rng.uniform(*noise_cfg["vignette_range"])
    image = apply_vignetting(image, vignette_str, rng)
    params_applied["vignette_strength"] = float(vignette_str)

    # 8. Intensity variation (gain drift)
    gain = rng.uniform(*noise_cfg["gain_range"])
    offset = rng.uniform(*noise_cfg["offset_range"])
    image = apply_intensity_variation(image, gain, offset)
    params_applied["gain"] = float(gain)
    params_applied["offset"] = float(offset)

    return image, params_applied


# =============================================================================
# Image Pair Generator
# =============================================================================

def generate_image_pair(pair_index, params=None, rng=None):
    """
    Generate one (reference, search) image pair with known ground truth.

    Pipeline:
      1. Generate a large DRAM layout (10000x10000) at 100x resolution
         using the v2 capacitor-body model
      2. Extract a 1000x1000 crop centered near the layout center -> reference
      3. Downsample the full layout 10x to 1000x1000 -> search image
      4. Apply independent SEM noise to each image
      5. Record where the reference appears in the search image

    Parameters
    ----------
    pair_index : int
    params : dict, optional
    rng : np.random.Generator, optional

    Returns
    -------
    reference : np.ndarray  (1000x1000 uint8)
    search : np.ndarray     (1000x1000 uint8)
    ground_truth : dict
    gen_params : dict
    """
    if rng is None:
        rng = np.random.default_rng()

    # Spawn independent RNGs for layout, ref noise, search noise
    child_rngs = rng.spawn(3)
    layout_rng = child_rngs[0]
    ref_noise_rng = child_rngs[1]
    search_noise_rng = child_rngs[2]

    # --- Step 1: Generate the 10000x10000 master DRAM layout ---
    master_size = 10000
    print(f"  [Pair {pair_index:03d}] Generating {master_size}x{master_size} master layout...")

    master_layout, defect_log = generate_dram_layout_v2(
        master_size, master_size, params=params, rng=layout_rng
    )

    # --- Step 2: Extract reference crop (1000x1000) from master ---
    ref_size = 1000
    center_y = master_size // 2
    center_x = master_size // 2

    # Drift offset aligned to 10px grid (avoids aliasing artifacts between
    # the reference crop and the 10x-downscaled search image)
    drift_y = rng.integers(-22, 22) * 10
    drift_x = rng.integers(-22, 22) * 10

    ref_y = center_y + drift_y - (ref_size // 2)
    ref_x = center_x + drift_x - (ref_size // 2)

    # Clamp to valid master bounds
    ref_y = max(0, min(master_size - ref_size, ref_y))
    ref_x = max(0, min(master_size - ref_size, ref_x))

    reference_clean = master_layout[ref_y:ref_y + ref_size,
                                    ref_x:ref_x + ref_size].copy()

    # --- Step 3: Build search image by downsampling the master ---
    search_size = 1000
    master_pil = Image.fromarray(master_layout.astype(np.uint8), mode='L')
    search_pil = master_pil.resize((search_size, search_size), Image.LANCZOS)
    search_clean = np.array(search_pil, dtype=np.float64)

    # --- Step 4: Compute ground truth coordinates ---
    gt_center_x = int(round((ref_x + ref_size / 2) / 10.0))
    gt_center_y = int(round((ref_y + ref_size / 2) / 10.0))
    gt_center_x = max(0, min(search_size - 1, gt_center_x))
    gt_center_y = max(0, min(search_size - 1, gt_center_y))

    # --- Step 5: Apply INDEPENDENT noise to each image ---
    ref_noise_cfg = {
        "blur_range": NOISE_PARAMS["blur_sigma_ref"],
        "edge_range": NOISE_PARAMS["edge_brightness_ref"],
        "poisson_range": NOISE_PARAMS["ref_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["ref_gaussian_std"],
        "beam_drift_range": (0.0, NOISE_PARAMS["beam_drift_amplitude"][1] * 0.3),
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": (0.0, 0.0),       # reference has no rotation
        "vignette_range": (0.03, 0.08),
        "gain_range": (0.95, 1.05),
        "offset_range": (-4, 4),
    }

    search_noise_cfg = {
        "blur_range": NOISE_PARAMS["blur_sigma_search"],
        "edge_range": NOISE_PARAMS["edge_brightness_search"],
        "poisson_range": NOISE_PARAMS["search_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["ref_gaussian_std"],
        "beam_drift_range": NOISE_PARAMS["beam_drift_amplitude"],
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": NOISE_PARAMS["rotation_range_deg"],
        "vignette_range": NOISE_PARAMS["vignette_strength"],
        "gain_range": NOISE_PARAMS["gain_variation"],
        "offset_range": NOISE_PARAMS["offset_variation"],
    }

    reference_noisy, ref_params = apply_full_sem_noise(
        reference_clean, ref_noise_cfg, ref_noise_rng
    )
    search_noisy, search_params = apply_full_sem_noise(
        search_clean, search_noise_cfg, search_noise_rng
    )

    reference_out = np.clip(reference_noisy, 0, 255).astype(np.uint8)
    search_out = np.clip(search_noisy, 0, 255).astype(np.uint8)

    ground_truth = {
        "center_x": int(gt_center_x),
        "center_y": int(gt_center_y),
        "ref_crop_x": int(ref_x),
        "ref_crop_y": int(ref_y),
        "ref_size": int(ref_size),
        "scale_factor": 10,
        "defects": defect_log,
    }

    gen_params = {
        "pair_index": int(pair_index),
        "master_size": int(master_size),
        "ref_noise_params": ref_params,
        "search_noise_params": search_params,
        "ground_truth": ground_truth,
    }

    return reference_out, search_out, ground_truth, gen_params


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Synthetic DRAM SEM Image Pair Generator (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_generator.py --style DRAM --num_pairs 50 --output_dir ./generated_data
  python dataset_generator.py --style DRAM --num_pairs 5 --output_dir ./test_data --seed 42
        """
    )
    parser.add_argument("--style", type=str, default="DRAM",
                        choices=["DRAM", "FinFET"],
                        help="Architecture style (default: DRAM)")
    parser.add_argument("--num_pairs", type=int, default=50,
                        help="Number of image pairs to generate (default: 50)")
    parser.add_argument("--output_dir", type=str, default="./generated_data",
                        help="Output directory (default: ./generated_data)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")

    args = parser.parse_args()

    if args.style != "DRAM":
        print("ERROR: Only DRAM-style generation is implemented.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time())
    master_rng = np.random.default_rng(seed)

    print("=" * 60)
    print("Drift-Sense: Synthetic DRAM Dataset Generator  (v2)")
    print("=" * 60)
    print(f"  Style:      {args.style}")
    print(f"  Pairs:      {args.num_pairs}")
    print(f"  Output:     {output_dir.resolve()}")
    print(f"  Seed:       {seed}")
    print("=" * 60)

    all_metadata = {
        "version": "2.0",
        "style": args.style,
        "num_pairs": args.num_pairs,
        "seed": seed,
        "pairs": [],
    }

    total_start = time.time()

    for i in range(1, args.num_pairs + 1):
        pair_start = time.time()

        reference, search, ground_truth, gen_params = generate_image_pair(
            pair_index=i, params=DRAM_PARAMS, rng=master_rng
        )

        pair_dir = output_dir / f"pair_{i:03d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(reference, mode='L').save(str(pair_dir / "reference.png"))
        Image.fromarray(search, mode='L').save(str(pair_dir / "search.png"))

        with open(pair_dir / "ground_truth.json", 'w') as f:
            json.dump(ground_truth, f, indent=2)

        pair_time = time.time() - pair_start

        pair_meta = {
            "pair_index": i,
            "ground_truth": ground_truth,
            "generation_time_sec": round(pair_time, 2),
        }
        all_metadata["pairs"].append(pair_meta)

        defect_count = len(ground_truth.get("defects", []))
        print(f"  [Pair {i:03d}] Done. GT=({ground_truth['center_x']}, "
              f"{ground_truth['center_y']}) | Defects: {defect_count} | Time: {pair_time:.1f}s")

    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"DONE! Generated {args.num_pairs} pairs in {total_time:.1f}s")
    print(f"Saved to: {output_dir.resolve()}")
    print(f"Metadata: {meta_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
