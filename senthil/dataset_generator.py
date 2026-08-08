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
import cv2
import gc
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
# Annular Ring Contact Layout Parameters (Geometric Pattern Model)
# =============================================================================

RING_PARAMS = {
    "image_size": 10000,
    "pitch": 160,                        # center-to-center distance of ring grid (px at 100x)
    "ring_radius": 52,                   # outer radius of bright annular ring (px)
    "ring_width": 12,                    # thickness of the bright ring wall (px)
    "ring_intensity": 220,               # intensity of bright annular ring (0-255)
    "ring_core_intensity": 90,           # intensity of ring interior core (0-255)
    "background_intensity": 75,          # uniform gray background intensity (0-255)
    "cross_width_v": 45,                 # vertical metal track width (px)
    "cross_width_h": 45,                 # horizontal metal track width (px)
    "cross_v_intensity": 160,            # light gray vertical track intensity
    "cross_h_intensity": 50,             # dark gray horizontal track intensity
    "center_pad_size": 160,              # central square pad side length (px)
    "center_pad_intensity": 30,          # dark central square pad intensity
    "corner_rounding_sigma": 2.5,        # lithography corner rounding blur sigma
    "line_edge_roughness": 1.2,          # line edge roughness RMS amplitude (px)
    "pitch_jitter": 0.8,                 # placement jitter (px RMS)
    "intensity_variation": 4.0,          # per-feature intensity variation
    "random_seed": None,
}


# =============================================================================
# SEM Noise Model Parameters
# =============================================================================
# Reference [2]: z = k * Poisson(x/k) + Gaussian(0, sigma^2)
# Calibrated to match real DRAM SEM images from dataset/

NOISE_PARAMS = {
    # Reference image (100x target -- ultra clear & clean)
    "ref_poisson_scale": (35.0, 60.0),   # high electron count => ultra low shot noise
    "ref_gaussian_std": (0.1, 0.8),      # negligible electronic noise

    # Search image (10x search -- high noise level as requested)
    "search_poisson_scale": (3.0, 7.0),  # heavy Poisson shot noise
    "search_gaussian_std": (8.0, 18.0),  # heavy Gaussian read noise

    # Edge brightening [1]
    "edge_brightness_ref": (0.12, 0.25),
    "edge_brightness_search": (0.04, 0.12),

    # Gaussian blur -- beam PSF [5]
    "blur_sigma_ref": (0.2, 0.6),        # sharp target focus
    "blur_sigma_search": (1.2, 2.5),     # search image defocus/blur

    # Rotation [1] -- stage drift between captures
    "rotation_range_deg": (-1.0, 1.0),

    # Vignetting -- cos^4 radial falloff [1]
    "vignette_strength": (0.15, 0.35),

    # Intensity variation -- detector gain drift
    "gain_variation": (0.85, 1.15),
    "offset_variation": (-15, 15),

    # Beam-current drift -- slow sinusoidal modulation of scan lines
    "beam_drift_amplitude": (0.02, 0.08), # scan-line intensity modulation
    "beam_drift_period": (80, 300),       # pixels
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
    layout = np.full((height, width), wall_int, dtype=np.float32)

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
    body_mask = (layout < (body_int + wall_int) / 2).astype(np.float32)
    ndimage.gaussian_filter(body_mask, sigma=corner_r * 0.4, output=body_mask)
    # Blend: layout = layout * (1 - body_mask) + body_int * body_mask
    layout += (body_int - layout) * body_mask
    del body_mask

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
# =============================================================================
# Annular Contact Array Layout Procedural Generator
# =============================================================================

def draw_annular_contacts(layout, width, height, pitch=160, ring_radius=52, ring_width=12,
                           ring_intensity=220, ring_core_intensity=80, background_intensity=25,
                           pitch_jitter=0.8, ler_amp=1.2, intensity_var=4.0, rng=None):
    """
    Draw a periodic square grid of alternating horizontal and vertical capsule/annular contact structures.
    Reproduces the woven DRAM semiconductor layout topology from SEM imagery.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Grid center coordinates
    grid_x = np.arange(pitch // 2, width + pitch, pitch, dtype=np.float64)
    grid_y = np.arange(pitch // 2, height + pitch, pitch, dtype=np.float64)

    # Capsule dimensions derived from ring_radius & ring_width
    cap_len = ring_radius * 1.35
    cap_wid = ring_radius * 0.65
    r_out = cap_wid / 2.0
    r_in = max(1.0, r_out - ring_width)

    for i, cy in enumerate(grid_y):
        for j, cx in enumerate(grid_x):
            # Apply pitch placement jitter
            cy_j = cy + rng.normal(0, pitch_jitter)
            cx_j = cx + rng.normal(0, pitch_jitter)
            cy_int, cx_int = int(round(cy_j)), int(round(cx_j))

            # Alternate orientation (checkerboard weave: horizontal vs vertical)
            is_horizontal = ((i + j) % 2 == 0)

            # Local ROI box
            box_r = int(cap_len + 12)
            y0, y1 = max(0, cy_int - box_r), min(height, cy_int + box_r)
            x0, x1 = max(0, cx_int - box_r), min(width, cx_int + box_r)

            if y1 <= y0 or x1 <= x0:
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            dy = yy - cy_int
            dx = xx - cx_int

            if is_horizontal:
                # Capsule shape oriented along X-axis
                dx_clamped = np.clip(np.abs(dx) - (cap_len / 2.0 - r_out), 0, None)
                dist = np.sqrt(dx_clamped**2 + dy**2)
            else:
                # Capsule shape oriented along Y-axis
                dy_clamped = np.clip(np.abs(dy) - (cap_len / 2.0 - r_out), 0, None)
                dist = np.sqrt(dx**2 + dy_clamped**2)

            ring_mask = (dist >= r_in) & (dist <= r_out)
            core_mask = dist < r_in

            local_ring_int = np.clip(ring_intensity + rng.normal(0, intensity_var), 0, 255)
            local_core_int = np.clip(ring_core_intensity + rng.normal(0, intensity_var), 0, 255)

            roi = layout[y0:y1, x0:x1]
            roi[core_mask] = local_core_int
            roi[ring_mask] = local_ring_int


def draw_center_cross(layout, width, height, cx=None, cy=None, cross_width=45, cross_intensity=230, rng=None):
    """
    Draw 4-way bright metal interconnect tracks crossing through layout at (cx, cy).
    """
    if cx is None: cx = width // 2
    if cy is None: cy = height // 2
    hw = cross_width // 2

    # Vertical metal track
    x0_v, x1_v = max(0, cx - hw), min(width, cx + hw)
    layout[:, x0_v:x1_v] = cross_intensity

    # Horizontal metal track
    y0_h, y1_h = max(0, cy - hw), min(height, cy + hw)
    layout[y0_h:y1_h, :] = cross_intensity


def draw_center_contact(layout, width, height, cx=None, cy=None, center_pad_size=160,
                        center_pad_intensity=20, border_width=12, border_intensity=220,
                        pad_shape="square", rng=None):
    """
    Draw central contact pad (black mark landmark) with outer bright border frame at (cx, cy).
    """
    if cx is None: cx = width // 2
    if cy is None: cy = height // 2

    pad_w = center_pad_size
    pad_h = int(center_pad_size * (rng.uniform(0.85, 1.15) if rng is not None else 1.0))

    hw_b = pad_w // 2
    hh_b = pad_h // 2

    y0_b, y1_b = max(0, cy - hh_b), min(height, cy + hh_b)
    x0_b, x1_b = max(0, cx - hw_b), min(width, cx + hw_b)

    if y1_b <= y0_b or x1_b <= x0_b:
        return

    layout[y0_b:y1_b, x0_b:x1_b] = border_intensity
    hw_i = max(1, hw_b - border_width)
    hh_i = max(1, hh_b - border_width)
    y0_i, y1_i = max(0, cy - hh_i), min(height, cy + hh_i)
    x0_i, x1_i = max(0, cx - hw_i), min(width, cx + hw_i)
    layout[y0_i:y1_i, x0_i:x1_i] = center_pad_intensity

    if pad_shape == "rounded":
        layout[y0_b:y1_b, x0_b:x1_b] = ndimage.gaussian_filter(layout[y0_b:y1_b, x0_b:x1_b], sigma=1.5)


def draw_stadium_capsules_row(layout, y_center, x_start, x_end, pitch=32, cap_len=45, cap_wid=22,
                              wall_thick=4, ring_int=220, core_int=70, rng=None):
    """
    Draw horizontal rows of stadium-shaped capsule slot contacts matching peripheral sense-amp rows.
    """
    h, w = layout.shape
    r_out = cap_wid / 2.0
    r_in = max(1.0, r_out - wall_thick)

    for cx in range(x_start + pitch // 2, x_end, pitch):
        cx_j = cx + (rng.normal(0, 0.5) if rng is not None else 0)
        cy_j = y_center + (rng.normal(0, 0.5) if rng is not None else 0)

        cy_int, cx_int = int(round(cy_j)), int(round(cx_j))
        box_r = int(cap_len // 2 + 8)

        y0, y1 = max(0, cy_int - box_r), min(h, cy_int + box_r)
        x0, x1 = max(0, cx_int - box_r), min(w, cx_int + box_r)

        if y1 <= y0 or x1 <= x0:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy = yy - cy_int
        dx = xx - cx_int

        dx_clamped = np.clip(np.abs(dx) - (cap_len / 2.0 - r_out), 0, None)
        dist = np.sqrt(dx_clamped**2 + dy**2)

        ring_mask = (dist >= r_in) & (dist <= r_out)
        core_mask = dist < r_in

        roi = layout[y0:y1, x0:x1]
        roi[core_mask] = core_int
        roi[ring_mask] = ring_int


def draw_circular_via_array(layout, x0, y0, x1, y1, pitch=42, via_r=16, wall_w=5,
                            ring_int=220, core_int=50, rng=None):
    """
    Draw matrix grid array of circular via contacts matching top-right / bottom-right peripheral blocks.
    """
    h, w = layout.shape
    r_out = via_r
    r_in = max(1.0, r_out - wall_w)

    for cy in range(y0 + pitch // 2, y1, pitch):
        for cx in range(x0 + pitch // 2, x1, pitch):
            cy_i, cx_i = int(cy), int(cx)
            box_r = r_out + 4
            vy0, vy1 = max(0, cy_i - box_r), min(h, cy_i + box_r)
            vx0, vx1 = max(0, cx_i - box_r), min(w, cx_i + box_r)
            if vy1 <= vy0 or vx1 <= vx0:
                continue

            yy, xx = np.ogrid[vy0:vy1, vx0:vx1]
            dist = np.sqrt((xx - cx_i)**2 + (yy - cy_i)**2)
            ring_mask = (dist >= r_in) & (dist <= r_out)
            core_mask = dist < r_in

            roi = layout[vy0:vy1, vx0:vx1]
            roi[core_mask] = core_int
            roi[ring_mask] = ring_int


def draw_vertical_line_grating(layout, x0, y0, x1, y1, pitch=24, line_w=12, line_int=200):
    """
    Draw dense parallel vertical metal line gratings matching peripheral interconnect buses.
    """
    hw = line_w // 2
    for cx in range(x0 + pitch // 2, x1, pitch):
        lx0 = max(0, cx - hw)
        lx1 = min(layout.shape[1], cx + hw)
        layout[y0:y1, lx0:lx1] = line_int


def apply_process_variation(layout, corner_rounding_sigma=2.5, line_edge_roughness=1.2, rng=None):
    """
    Apply SEM process variations: lithography corner rounding blur and Line Edge Roughness (LER).
    Ultra memory-efficient slice execution.
    """
    if corner_rounding_sigma > 0:
        ndimage.gaussian_filter(layout, sigma=corner_rounding_sigma, output=layout)
    if line_edge_roughness > 0 and rng is not None:
        chunk_size = 500
        for r0 in range(0, layout.shape[0], chunk_size):
            r1 = min(layout.shape[0], r0 + chunk_size)
            chunk_float = layout[r0:r1].astype(np.float32)
            noise = rng.standard_normal((r1 - r0, layout.shape[1]), dtype=np.float32)
            noise *= line_edge_roughness
            chunk_float += noise
            np.clip(chunk_float, 0, 255, out=chunk_float)
            layout[r0:r1] = chunk_float.astype(np.uint8)
            del chunk_float, noise
        gc.collect()
    return layout


def generate_ring_array_layout(image_size=10000, pitch=160, ring_radius=52, ring_width=12,
                               ring_intensity=220, background_intensity=25, cross_width=45,
                               center_pad_size=160, corner_rounding_sigma=2.5,
                               line_edge_roughness=1.2, pitch_jitter=0.8,
                               intensity_variation=4.0, random_seed=None, params=None, rng=None):
    """
    Generate complete multi-zone hierarchical DRAM semiconductor wafer layout procedurally.
    Matches the photorealistic reference wafer image:
      - Sub-array block grid with sense-amp & SWD boundary bands
      - Double rows of stadium/capsule slot contacts in top/bottom peripheral bands
      - Top-right & Bottom-right circular via contact arrays
      - Middle-right & left vertical line grating buses
      - 2 or 3 black mark landmarks with 4-way interconnect cross
    """
    if rng is None:
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        elif params is not None and params.get("random_seed") is not None:
            rng = np.random.default_rng(params.get("random_seed"))
        else:
            rng = np.random.default_rng()

    gc.collect()
    width = height = image_size

    # --- 1. Background (uint8, 95MB memory footprint) ---
    layout = np.full((height, width), background_intensity, dtype=np.uint8)

    # --- 2. Central Field (Sub-Array Blocks) ---
    # Draw dense sub-array cell grid in central field [800..8500 x 2000..7800]
    draw_annular_contacts(layout, width, height, pitch=pitch, ring_radius=ring_radius,
                          ring_width=ring_width, ring_intensity=ring_intensity,
                          background_intensity=background_intensity,
                          pitch_jitter=pitch_jitter, ler_amp=line_edge_roughness,
                          intensity_var=intensity_variation, rng=rng)

    # Sub-Array Block double-line boundaries (SWD columns and Sense-Amp rows)
    block_px, block_py = 1800, 1500
    for bx in range(800 + block_px, 8500, block_px):
        layout[2000:7800, max(0, bx - 8):min(width, bx + 8)] = 210
        layout[2000:7800, max(0, bx + 20):min(width, bx + 36)] = 210

    for by in range(2000 + block_py, 7800, block_py):
        layout[max(0, by - 8):min(height, by + 8), 800:8500] = 210
        layout[max(0, by + 20):min(height, by + 36), 800:8500] = 210

    # --- 3. Top & Bottom Peripheral Bands (Stadium Capsule Rows & Gratings) ---
    for band_y in [1200, 1400, 8200, 8400]:
        draw_stadium_capsules_row(layout, y_center=band_y, x_start=800, x_end=8500,
                                   pitch=34, cap_len=45, cap_wid=22, wall_thick=4, rng=rng)

    # Vertical line gratings above/below capsule rows
    draw_vertical_line_grating(layout, 800, 600, 8500, 1100, pitch=24, line_w=10, line_int=180)
    draw_vertical_line_grating(layout, 800, 8550, 8500, 9400, pitch=24, line_w=10, line_int=180)

    # --- 4. Right Peripheral Blocks ---
    # Top-Right Circular Via Array
    draw_circular_via_array(layout, 8500, 0, 10000, 2000, pitch=42, via_r=16, rng=rng)
    # Bottom-Right Circular Via Array
    draw_circular_via_array(layout, 8500, 7800, 10000, 10000, pitch=42, via_r=16, rng=rng)
    # Middle-Right Vertical Line Grating
    draw_vertical_line_grating(layout, 8500, 2000, 10000, 7800, pitch=24, line_w=12, line_int=190)

    # --- 5. Left Vertical Margin Bus Strip ---
    draw_vertical_line_grating(layout, 0, 0, 800, 10000, pitch=30, line_w=14, line_int=195)
    layout[:, 780:800] = 230  # Margin line

    # --- 6. Multi-Landmark Black Mark Placement (2 or 3 places) ---
    rand_pad_size = int(rng.integers(140, 240))
    rand_pad_intensity = float(rng.uniform(10, 35))
    rand_border_width = int(rng.integers(10, 24))
    rand_border_intensity = float(rng.uniform(210, 255))
    rand_pad_shape = rng.choice(["square", "rounded"])

    num_landmarks = int(rng.integers(2, 4)) # 2 or 3 places
    landmarks = []

    # Landmark 1 (near center region)
    c1_x = int(round((width * 0.5 + rng.integers(-int(width * 0.12), int(width * 0.12))) / pitch) * pitch + pitch // 2)
    c1_y = int(round((height * 0.5 + rng.integers(-int(height * 0.12), int(height * 0.12))) / pitch) * pitch + pitch // 2)
    landmarks.append((c1_x, c1_y))

    # Landmark 2 (offset far enough inside bounds)
    c2_x = int(round((width * 0.5 + rng.choice([-1, 1]) * rng.integers(int(width * 0.20), int(width * 0.30))) / pitch) * pitch + pitch // 2)
    c2_y = int(round((height * 0.5 + rng.choice([-1, 1]) * rng.integers(int(height * 0.20), int(height * 0.30))) / pitch) * pitch + pitch // 2)
    c2_x = max(pitch * 2, min(width - pitch * 2, c2_x))
    c2_y = max(pitch * 2, min(height - pitch * 2, c2_y))
    landmarks.append((c2_x, c2_y))

    if num_landmarks >= 3:
        c3_x = int(round((width * 0.5 + rng.choice([-1, 1]) * rng.integers(int(width * 0.22), int(width * 0.32))) / pitch) * pitch + pitch // 2)
        c3_y = int(round((height * 0.5 + rng.choice([-1, 1]) * rng.integers(int(height * 0.22), int(height * 0.32))) / pitch) * pitch + pitch // 2)
        c3_x = max(pitch * 2, min(width - pitch * 2, c3_x))
        c3_y = max(pitch * 2, min(height - pitch * 2, c3_y))
        landmarks.append((c3_x, c3_y))

    # Draw 4-way metal interconnect cross & black mark pad at all landmark locations
    for (cx, cy) in landmarks:
        draw_center_cross(layout, width, height, cx=cx, cy=cy, cross_width=cross_width, rng=rng)
        draw_center_contact(layout, width, height, cx=cx, cy=cy,
                            center_pad_size=rand_pad_size,
                            center_pad_intensity=rand_pad_intensity,
                            border_width=rand_border_width,
                            border_intensity=rand_border_intensity,
                            pad_shape=rand_pad_shape, rng=rng)

    # --- 7. Apply Process Variations ---
    layout = apply_process_variation(layout, corner_rounding_sigma=corner_rounding_sigma,
                                     line_edge_roughness=line_edge_roughness, rng=rng)

    defect_log = [{
        "center_pad_size": rand_pad_size,
        "landmarks": landmarks,
        "center_pad_shape": str(rand_pad_shape)
    }]
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
    noisy = rng.poisson(lam).astype(np.float32) / scale
    return np.clip(noisy, 0, 255)


def apply_gaussian_noise(image, std, rng):
    """
    Apply Gaussian (read/electronic) noise.

    Signal-independent additive white Gaussian noise from SEM detector
    electronics (amplifiers, ADC quantization).

    References: [2] Foi et al., 2008; [1] Goldstein et al., 2018.
    """
def apply_gaussian_noise(image, std, rng):
    if std <= 0:
        return image
    noise = rng.normal(0, std, size=image.shape).astype(np.float32)
    img_f32 = image.astype(np.float32)
    np.add(img_f32, noise, out=img_f32)
    return np.clip(img_f32, 0.0, 255.0)


def apply_edge_brightening(image, strength):
    """
    Apply SEM-style edge brightening (topographic contrast) using OpenCV Sobel.
    """
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    max_val = grad.max()
    if max_val > 0:
        grad /= max_val
    return np.clip(image + strength * grad * 80.0, 0, 255)


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
    vignette_map = np.clip(1.0 - strength * (dist ** 2), 0.4, 1.0).astype(np.float32)
    return np.clip(cv2.multiply(image.astype(np.float32), vignette_map), 0, 255)


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
    """
    if rng is None:
        rng = np.random.default_rng()

    params_applied = {}

    blur_sigma = rng.uniform(*noise_cfg["blur_range"])
    image = apply_gaussian_blur(image, blur_sigma)
    params_applied["blur_sigma"] = float(blur_sigma)

    edge_str = rng.uniform(*noise_cfg["edge_range"])
    image = apply_edge_brightening(image, edge_str)
    params_applied["edge_brightness"] = float(edge_str)

    drift_amp = rng.uniform(*noise_cfg["beam_drift_range"])
    drift_period = rng.uniform(*noise_cfg["beam_drift_period_range"])
    image = apply_beam_drift(image, drift_amp, drift_period, rng)
    params_applied["beam_drift_amplitude"] = float(drift_amp)

    poisson_scale = rng.uniform(*noise_cfg["poisson_range"])
    image = apply_poisson_noise(image, poisson_scale, rng)
    params_applied["poisson_scale"] = float(poisson_scale)

    gauss_std = rng.uniform(*noise_cfg["gaussian_range"])
    image = apply_gaussian_noise(image, gauss_std, rng)
    params_applied["gaussian_std"] = float(gauss_std)

    rotation_deg = rng.uniform(*noise_cfg["rotation_range"])
    image = apply_rotation(image, rotation_deg)
    params_applied["rotation_deg"] = float(rotation_deg)

    vignette_str = rng.uniform(*noise_cfg["vignette_range"])
    image = apply_vignetting(image, vignette_str, rng)
    params_applied["vignette_strength"] = float(vignette_str)

    gain = rng.uniform(*noise_cfg["gain_range"])
    offset = rng.uniform(*noise_cfg["offset_range"])
    image = apply_intensity_variation(image, gain, offset)
    params_applied["gain"] = float(gain)
    params_applied["offset"] = float(offset)

    return image, params_applied


def generate_concentric_ring_search_image(width=1000, height=1000, rng=None):
    """
    Generate 1000x1000 search image matching concentric annular ring SEM wafer layout.
      - Grid of concentric ring contacts (pitch = 32 px at 10x search scale).
      - Unique target contact (at gt_x, gt_y) has a dominant high-contrast dark square notch mark
        with a bright white rim frame embedded inside its concentric ring wall.
    """
    if rng is None:
        rng = np.random.default_rng()

    layout = np.full((height, width), 40, dtype=np.uint8)
    pitch = 32

    # Target position in search image
    gt_x = int(rng.integers(350, 650))
    gt_y = int(rng.integers(350, 650))
    gt_x = int(round(gt_x / pitch)) * pitch
    gt_y = int(round(gt_y / pitch)) * pitch

    # 1. Concentric Annular Ring Contact Grid
    for cy in range(pitch, height - pitch + 1, pitch):
        for cx in range(pitch, width - pitch + 1, pitch):
            # Outer ring wall (radius 14 px, thickness 3 px)
            cv2.circle(layout, (cx, cy), 14, (210,), 3)
            # Inner ring wall (radius 8 px, thickness 2 px)
            cv2.circle(layout, (cx, cy), 8, (230,), 2)
            # Center dark core (radius 3 px)
            cv2.circle(layout, (cx, cy), 3, (30,), -1)

    # 2. Embed High-Contrast Dominant Unique Notch Mark on Target Contact (16x16 px)
    notch_w = 12
    nx = gt_x - 6
    ny = gt_y - 6

    # Bright white outer rim frame (intensity 255)
    cv2.rectangle(layout, (nx - 3, ny - 3), (nx + notch_w + 3, ny + notch_w + 3), (255,), 2)
    # Deep black inner notch core (intensity 0)
    layout[ny : ny + notch_w, nx : nx + notch_w] = 0

    return layout, (gt_x, gt_y)


def generate_concentric_ring_reference_crop(crop_w=1000, crop_h=1000, rng=None):
    """
    Generate 1000x1000 reference target crop at 100x high magnification centered on unique ring contact.
      - Features drawn 10x larger (pitch = 320 px, outer ring radius = 140 px, inner ring radius = 80 px).
      - Contains dominant high-contrast dark square notch mark (120x120 px) with bright white rim.
    """
    if rng is None:
        rng = np.random.default_rng()

    ref = np.full((crop_h, crop_w), 40, dtype=np.uint8)

    cx = crop_w // 2
    cy = crop_h // 2
    pitch_100x = 320

    # Draw Concentric Ring Contact Grid at 100x scale
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            rcx = cx + dx * pitch_100x
            rcy = cy + dy * pitch_100x

            if -200 <= rcx <= crop_w + 200 and -200 <= rcy <= crop_h + 200:
                # Outer ring wall (radius 140 px, thickness 30 px)
                cv2.circle(ref, (rcx, rcy), 140, (210,), 30)
                # Inner ring wall (radius 80 px, thickness 20 px)
                cv2.circle(ref, (rcx, rcy), 80, (230,), 20)
                # Center dark core (radius 30 px)
                cv2.circle(ref, (rcx, rcy), 30, (30,), -1)

    # Embed High-Contrast Dominant Unique Notch Mark on Center Target Contact (120x120 px)
    notch_w_100x = 120
    nx = cx - 60
    ny = cy - 60

    # Bright white outer rim frame (intensity 255)
    cv2.rectangle(ref, (nx - 30, ny - 30), (nx + notch_w_100x + 30, ny + notch_w_100x + 30), (255,), 20)
    # Deep black inner notch core (intensity 0)
    ref[ny : ny + notch_w_100x, nx : nx + notch_w_100x] = 0

    ref_float = ref.astype(np.float32)
    ndimage.gaussian_filter(ref_float, sigma=0.6, output=ref_float)
    return np.clip(ref_float, 0, 255).astype(np.uint8)


def apply_sem_physics(image, noise_cfg, rng):
    return apply_full_sem_noise(image, noise_cfg, rng)


def generate_image_pair(pair_index, style="RING", params=None, rng=None):
    """
    Generate one (reference, search) image pair matching Applied Materials Drift-Sense spec:
      - Reference Image: 1000x1000 px at 100x high magnification (centered on unique ring contact).
      - Search Image: 1000x1000 px at 10x low magnification (concentric ring array with unique target).
      - Ground Truth: exact center (gt_x, gt_y) of target ring in search image.
    """
    gc.collect()
    if rng is None:
        rng = np.random.default_rng()

    child_rngs = rng.spawn(3)
    layout_rng = child_rngs[0]
    ref_noise_rng = child_rngs[1]
    search_noise_rng = child_rngs[2]

    # 1. Generate clean 10x search layout + target coordinates
    search_clean, (gt_x, gt_y) = generate_concentric_ring_search_image(1000, 1000, rng=layout_rng)

    # 2. Generate clean 100x high-magnification reference target crop
    ref_clean = generate_concentric_ring_reference_crop(1000, 1000, rng=layout_rng).astype(np.float32)
    search_clean = search_clean.astype(np.float32)

    # 3. Apply independent SEM noise degradation physics
    ref_noise_cfg = {
        "blur_range": NOISE_PARAMS["blur_sigma_ref"],
        "edge_range": NOISE_PARAMS["edge_brightness_ref"],
        "poisson_range": NOISE_PARAMS["ref_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["ref_gaussian_std"],
        "beam_drift_range": (0.0, NOISE_PARAMS["beam_drift_amplitude"][1] * 0.3),
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": (0.0, 0.0),
        "vignette_range": (0.03, 0.08),
        "gain_range": (0.95, 1.05),
        "offset_range": (-4, 4),
    }

    search_noise_cfg = {
        "blur_range": NOISE_PARAMS["blur_sigma_search"],
        "edge_range": NOISE_PARAMS["edge_brightness_search"],
        "poisson_range": NOISE_PARAMS["search_poisson_scale"],
        "gaussian_range": NOISE_PARAMS["search_gaussian_std"],
        "beam_drift_range": NOISE_PARAMS["beam_drift_amplitude"],
        "beam_drift_period_range": NOISE_PARAMS["beam_drift_period"],
        "rotation_range": NOISE_PARAMS["rotation_range_deg"],
        "vignette_range": NOISE_PARAMS["vignette_strength"],
        "gain_range": NOISE_PARAMS["gain_variation"],
        "offset_range": NOISE_PARAMS["offset_variation"],
    }

    reference_out, ref_params = apply_sem_physics(ref_clean, ref_noise_cfg, ref_noise_rng)
    search_out, search_params = apply_sem_physics(search_clean, search_noise_cfg, search_noise_rng)

    reference_out = np.clip(reference_out, 0, 255).astype(np.uint8)
    search_out = np.clip(search_out, 0, 255).astype(np.uint8)

    ground_truth = {
        "center_x": int(gt_x),
        "center_y": int(gt_y),
        "downsample_factor": 10.0,
        "unique_notch": {
            "x": int(gt_x - 6),
            "y": int(gt_y - 4),
            "size": 6
        }
    }

    gen_params = {
        "pair_index": int(pair_index),
        "style": style,
        "ref_noise_params": ref_params,
        "search_noise_params": search_params,
        "ground_truth": ground_truth,
    }

    return reference_out, search_out, ground_truth, gen_params


# =============================================================================
# CLI Entry Point
def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Synthetic SEM Image Pair Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"Examples: python dataset_generator.py --style RING --num_pairs 50 --output_dir ./generated_data"
    )
    parser.add_argument("--style", type=str, default="RING",
                        choices=["RING", "DRAM", "FinFET"],
                        help="Architecture style (default: RING)")
    parser.add_argument("--num_pairs", type=int, default=50,
                        help="Number of image pairs to generate (default: 50)")
    parser.add_argument("--output_dir", type=str, default="./generated_data",
                        help="Output directory (default: ./generated_data)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: random)")

    args = parser.parse_args()

    if args.style not in ["RING", "DRAM"]:
        print(f"ERROR: Style '{args.style}' is not implemented. Use RING or DRAM.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(time.time())
    master_rng = np.random.default_rng(seed)

    print("=" * 60)
    print(f"Drift-Sense: Synthetic SEM Dataset Generator ({args.style})")
    print("=" * 60)
    print(f"  Style:      {args.style}")
    print(f"  Pairs:      {args.num_pairs}")
    print(f"  Output:     {output_dir.resolve()}")
    print(f"  Seed:       {seed}")
    print("=" * 60)

    all_metadata = {
        "version": "2.5",
        "style": args.style,
        "num_pairs": args.num_pairs,
        "seed": seed,
        "pairs": [],
    }

    total_start = time.time()

    for i in range(1, args.num_pairs + 1):
        pair_start = time.time()

        reference, search, ground_truth, gen_params = generate_image_pair(
            pair_index=i, style=args.style, params=None, rng=master_rng
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

