#!/usr/bin/env python3
"""
EXTREME TOUGH SEM Dataset Generator
===================================

Generates extremely challenging 100 SEM image pairs with:
- Maximum degradations (4-5 per image vs standard 2)
- Extreme parameter ranges
- Complex noise combinations
- RGB and grayscale mixed
- Aggressive geometric distortions
- Severe acquisition artifacts
- High hard-negative rate (40% vs standard 25%)

This is designed to be a STRESS TEST for localization algorithms.
"""

import os
import json
import cv2
import numpy as np
import gc
from pathlib import Path

# Import existing generator components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model", "standalone_dataset_generator"))
from master_generator_v2 import discover_render_functions, generate_clean_pair


# =============================================================================
# EXTREME DEGRADATION CONFIGURATION
# =============================================================================

EXTREME_CONFIG = {
    "enabled": True,
    "probability_applied": 1.0,    # 100% get degradation (no clean images)
    "min_total_models": 4,           # MINIMUM 4 degradations per image
    "max_total_models": 5,          # MAXIMUM 5 degradations per image
    
    # Extreme model weights (favoring challenging degradations)
    "model_weights": {
        # Category A: SEM Noise (extreme ranges)
        "poisson":            0.18,
        "gaussian":           0.18,
        "gamma":              0.10,
        "secondary_electron": 0.10,
        "low_frequency":      0.08,
        
        # Category B: Robustness Noise (higher probability)
        "speckle":            0.12,
        "banding":            0.10,
        "salt_pepper":        0.08,
        
        # Category C: Blur (severe ranges)
        "gaussian_blur":      0.12,
        "motion_blur":        0.08,
        
        # Category D: Geometric (aggressive parameters) - use actual function names
        "rotation":           0.05,
        "barrel":             0.04,
        "pincushion":         0.04,
        "projective":         0.03,
        
        # Category E: Acquisition (severe artifacts) - use actual function names
        "field_nonuniformity": 0.08,
        "vibration":           0.06,
        "stage_drift":         0.05,
        
        # Custom extreme degradations
        "extreme_combined_noise": 0.05,
        "extreme_pattern_corruption": 0.05,
    },
    
    # EXTREME strength ranges (much more challenging)
    "strength_ranges": {
        "poisson":            {"scale_min": 5.0, "scale_max": 15.0},           # Was 1.0-5.0
        "gaussian":           {"sigma_min": 10.0, "sigma_max": 30.0},         # Was 3.0-15.0
        "gamma":              {"shape_min": 20.0, "shape_max": 100.0},        # Was 10.0-50.0
        "secondary_electron": {"sigma_min": 8.0, "sigma_max": 25.0},          # Was 3.0-12.0
        "low_frequency":      {"amplitude_min": 15.0, "amplitude_max": 50.0,  # Was 5.0-20.0
                              "frequency_min": 1, "frequency_max": 5},
        
        "speckle":            {"sigma_min": 0.08, "sigma_max": 0.20},          # Was 0.02-0.08
        "banding":            {"amplitude_min": 10.0, "amplitude_max": 30.0},  # Was 3.0-12.0
        "salt_pepper":        {"density_min": 0.005, "density_max": 0.02},    # Was 0.0005-0.005
        
        "gaussian_blur":      {"ksize_min": 5, "ksize_max": 9,                # Was 3-5
                              "sigma_min": 1.5, "sigma_max": 3.5},             # Was 0.5-1.5
        "motion_blur":        {"ksize_min": 5, "ksize_max": 15,               # Was 3-7
                              "angle_min": 0.0, "angle_max": 180.0},
        
        "rotation":           {"angle_min": -5.0, "angle_max": 5.0},           # Was -2.0 to 2.0
        "barrel":             {"k1_min": 0.03, "k1_max": 0.08},               # Was 0.005-0.03
        "pincushion":         {"k1_min": -0.08, "k1_max": -0.03},             # Was -0.03 to -0.005
        "projective":         {"max_shift_min": 10, "max_shift_max": 30},      # Was 3-12
        
        "field_nonuniformity": {"strength_min": 0.20, "strength_max": 0.50},  # Was 0.05-0.20
        "vibration":           {"amplitude_min": 3, "amplitude_max": 8},      # Was 1-3
        "stage_drift":         {"dx_min": -15, "dx_max": 15,                   # Was -5 to 5
                              "dy_min": -15, "dy_max": 15},
        
        # Custom extreme degradations
        "extreme_combined_noise": {"sigma_min": 5.0, "sigma_max": 15.0, "scale_min": 5.0, "scale_max": 15.0},
        "extreme_pattern_corruption": {},
    },
}


# =============================================================================
# EXTREME DEGRADATION FUNCTIONS (Import and extend)
# =============================================================================

# Import standard degradation functions
from degradation_engine import (
    noise_poisson, noise_gaussian, noise_gamma, noise_secondary_electron, noise_low_frequency,
    noise_speckle, noise_banding, noise_salt_pepper,
    blur_gaussian, blur_motion,
    geo_rotation, geo_barrel, geo_pincushion, geo_projective,
    acq_field_nonuniformity, acq_vibration, acq_stage_drift
)

# Add new extreme degradation functions
def extreme_combined_noise(img_f32, rng, cfg):
    """Combines multiple noise types for maximum challenge"""
    # Add gaussian + poisson together
    sigma = rng.uniform(5.0, 15.0)
    scale = rng.uniform(5.0, 15.0)
    
    gaussian_noise = rng.normal(0, sigma, img_f32.shape).astype(np.float32)
    img_scaled = np.clip(img_f32 / 255.0 * scale, 0, None)
    poisson_noise = (rng.poisson(img_scaled).astype(np.float32) / scale * 255.0) - img_f32
    
    combined = img_f32 + gaussian_noise + poisson_noise
    return combined, {"model": "extreme_combined_noise", "sigma": round(float(sigma), 3), "scale": round(float(scale), 3)}

def extreme_pattern_corruption(img_f32, rng, cfg):
    """Adds structured pattern corruption that breaks periodicity"""
    h, w = img_f32.shape[:2]
    corruption_mask = np.zeros_like(img_f32)
    
    # Add random structured corruption blocks
    num_blocks = rng.integers(3, 8)
    for _ in range(num_blocks):
        bh = rng.integers(20, 100)
        bw = rng.integers(20, 100)
        by = rng.integers(0, h - bh)
        bx = rng.integers(0, w - bw)
        
        # Different corruption types
        corruption_type = rng.choice(['invert', 'noise', 'shift'])
        if corruption_type == 'invert':
            corruption_mask[by:by+bh, bx:bx+bw] = 255
        elif corruption_type == 'noise':
            corruption_mask[by:by+bh, bx:bx+bw] = rng.normal(0, 50, (bh, bw)).astype(np.float32)
        elif corruption_type == 'shift':
            shift = rng.integers(-30, 30)
            if img_f32.ndim == 2:
                corruption_mask[by:by+bh, bx:bx+bw] = shift
            else:
                corruption_mask[by:by+bh, bx:bx+bw, :] = shift
    
    return img_f32 + corruption_mask, {"model": "extreme_pattern_corruption", "blocks": num_blocks}


# =============================================================================
# EXTREME DEGRADATION ENGINE
# =============================================================================

DEGRADATION_FUNCTIONS = {
    "poisson": noise_poisson,
    "gaussian": noise_gaussian,
    "gamma": noise_gamma,
    "secondary_electron": noise_secondary_electron,
    "low_frequency": noise_low_frequency,
    "speckle": noise_speckle,
    "banding": noise_banding,
    "salt_pepper": noise_salt_pepper,
    "gaussian_blur": blur_gaussian,
    "motion_blur": blur_motion,
    "rotation": geo_rotation,
    "barrel": geo_barrel,
    "pincushion": geo_pincushion,
    "projective": geo_projective,
    "field_nonuniformity": acq_field_nonuniformity,
    "vibration": acq_vibration,
    "stage_drift": acq_stage_drift,
    "extreme_combined_noise": extreme_combined_noise,
    "extreme_pattern_corruption": extreme_pattern_corruption,
}

def apply_extreme_degradation(img_clean, gt_x, gt_y, rng, config):
    """
    Apply extreme degradations to create challenging test samples.
    Returns: (degraded_img, final_gt_x, final_gt_y, degradation_report)
    """
    if not config.get("enabled", True):
        return img_clean.copy(), gt_x, gt_y, []
    
    # Determine number of degradations (4-5 for extreme challenge)
    min_models = config.get("min_total_models", 4)
    max_models = config.get("max_total_models", 5)
    num_models = rng.integers(min_models, max_models + 1)
    
    # Select degradation models
    model_weights = config.get("model_weights", {})
    model_names = list(model_weights.keys())
    weights = list(model_weights.values())
    weights = np.array(weights) / np.sum(weights)
    
    selected_models = rng.choice(model_names, size=num_models, p=weights, replace=False)
    
    # Apply degradations sequentially
    img_degraded = img_clean.astype(np.float32).copy()
    degradation_report = []
    
    # Initialize GT coordinates (will be updated by geometric functions)
    final_gt_x, final_gt_y = gt_x, gt_y
    
    # Functions that require gt_x, gt_y parameters
    geo_acq_functions = ["rotation", "barrel", "pincushion", "projective", "vibration", "stage_drift"]
    
    for model_name in selected_models:
        if model_name not in DEGRADATION_FUNCTIONS:
            continue
            
        func = DEGRADATION_FUNCTIONS[model_name]
        strength_cfg = config.get("strength_ranges", {}).get(model_name, {})
        
        try:
            if model_name in geo_acq_functions:
                # These functions take gt_x, gt_y and return modified GT
                img_degraded, new_gt_x, new_gt_y, model_info = func(img_degraded.astype(np.uint8), final_gt_x, final_gt_y, rng, strength_cfg)
                img_degraded = img_degraded.astype(np.float32)
                final_gt_x, final_gt_y = new_gt_x, new_gt_y
            else:
                img_degraded, model_info = func(img_degraded, rng, strength_cfg)
            degradation_report.append(model_info)
        except Exception as e:
            print(f"Warning: Failed to apply {model_name}: {e}")
            continue
    
    # Clip to valid range
    img_degraded = np.clip(img_degraded, 0, 255).astype(np.uint8)
    
    return img_degraded, final_gt_x, final_gt_y, degradation_report


# =============================================================================
# EXTREME DATASET GENERATOR
# =============================================================================

def generate_extreme_tough_dataset(num_pairs=100, output_dir="./extreme_tough_dataset", master_seed=999):
    """
    Generate extremely challenging SEM dataset for stress testing.
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing architecture generators
    scripts_dir = os.path.join(os.path.dirname(__file__), "model", "standalone_dataset_generator", "clean_60_scripts")
    registry = discover_render_functions(scripts_dir)
    num_arch = len(registry)
    
    if num_arch == 0:
        raise RuntimeError(f"No generator scripts found in {scripts_dir}")
    
    # Use all architectures for maximum variety
    master_rng = np.random.default_rng(master_seed)
    indices = np.arange(num_arch)
    master_rng.shuffle(indices)
    
    print("=" * 80)
    print(f"EXTREME TOUGH SEM DATASET GENERATOR ({num_pairs} PAIRS)")
    print("=" * 80)
    print(f"Output Directory: {output_dir}")
    print(f"Total Generators: {num_arch}")
    print(f"Degradations per image: 4-5 (EXTREME)")
    print(f"Hard negative rate: 40% (vs standard 25%)")
    print(f"RGB samples: 30% (30 RGB, 70 grayscale)")
    print("=" * 80)
    
    manifest_entries = []
    t0 = time.time()
    
    for i in range(num_pairs):
        pair_id = f"pair_{i+1:04d}"
        pair_dir = os.path.join(output_dir, pair_id)
        os.makedirs(pair_dir, exist_ok=True)
        
        gt_json = os.path.join(pair_dir, "groundtruth.json")
        search_png = os.path.join(pair_dir, "search.png")
        target_png = os.path.join(pair_dir, "target.png")
        
        # Check if already exists
        if os.path.exists(gt_json) and os.path.exists(search_png) and os.path.exists(target_png):
            with open(gt_json, "r") as f:
                gt_info = json.load(f)
            manifest_entries.append(gt_info)
            print(f"  [Existing] [{i+1}/{num_pairs}] {pair_id}")
            continue
        
        # Select architecture
        arch_idx = indices[i % num_arch]
        arch = registry[arch_idx]
        folder_num = arch["folder_num"]
        render_fn = arch["render_fn"]
        generate_fn = arch["generate_fn"]
        
        render_seed = int(master_rng.integers(0, 2**31))
        degrade_seed = int(master_rng.integers(0, 2**31))
        degrade_rng = np.random.default_rng(degrade_seed)
        
        # EXTREME hard negative rate (40% vs standard 25%)
        is_extreme_hard_neg = (i % 5 < 2)  # 40% chance
        
        # Generate random coordinates across full valid range
        base_x = int(degrade_rng.integers(100, 900))
        base_y = int(degrade_rng.integers(100, 900))
        
        if is_extreme_hard_neg:
            # Extreme periodic shifts (larger than standard)
            dx_shift = int(degrade_rng.choice([50, -50, 75, -75, 100, -100]))
            dy_shift = int(degrade_rng.choice([70, -70, 50, -50, 85, -85]))
            gt_x = max(100, min(900, base_x + dx_shift))
            gt_y = max(100, min(900, base_y + dy_shift))
        else:
            gt_x, gt_y = base_x, base_y
        
        # Generate clean pair
        if render_fn is not None:
            search_clean, target_clean = generate_clean_pair(render_fn, gt_x, gt_y, render_seed)
        else:
            generate_fn(output_dir=pair_dir, seed=render_seed)
            s_path = os.path.join(pair_dir, "search.png")
            t_path = os.path.join(pair_dir, "target.png")
            if os.path.exists(s_path):
                search_clean = cv2.imread(s_path, cv2.IMREAD_UNCHANGED)
                target_clean = cv2.imread(t_path, cv2.IMREAD_UNCHANGED)
            else:
                search_clean = np.full((1000, 1000), 40, dtype=np.uint8)
                target_clean = np.full((1000, 1000), 40, dtype=np.uint8)
        
        # Determine if RGB (30% chance)
        is_rgb = (i % 10 < 3)  # 30% RGB
        
        # Ensure target is grayscale first
        if len(target_clean.shape) == 3:
            target_clean = cv2.cvtColor(target_clean, cv2.COLOR_BGR2GRAY)
        if len(search_clean.shape) == 3:
            search_clean = cv2.cvtColor(search_clean, cv2.COLOR_BGR2GRAY)
        
        # Save target/reference (always clean)
        if is_rgb:
            target_rgb = cv2.cvtColor(target_clean, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(pair_dir, "target.png"), target_rgb)
            cv2.imwrite(os.path.join(pair_dir, "reference.png"), target_rgb)
        else:
            cv2.imwrite(os.path.join(pair_dir, "target.png"), target_clean)
            cv2.imwrite(os.path.join(pair_dir, "reference.png"), target_clean)
        
        # Apply extreme degradations to search image
        search_degraded, final_gt_x, final_gt_y, report = apply_extreme_degradation(
            search_clean, gt_x, gt_y, degrade_rng, EXTREME_CONFIG
        )
        
        # Convert report to JSON-serializable format
        json_report = []
        for item in report:
            json_item = {}
            for key, value in item.items():
                if isinstance(value, (np.integer, np.floating)):
                    json_item[key] = float(value) if isinstance(value, np.floating) else int(value)
                else:
                    json_item[key] = value
            json_report.append(json_item)
        report = json_report
        
        # Convert to RGB if needed
        if is_rgb:
            search_degraded = cv2.cvtColor(search_degraded, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(os.path.join(pair_dir, "search.png"), search_degraded)
        
        # Cleanup
        del search_clean, target_clean, search_degraded
        if i % 20 == 0:
            gc.collect()
        
        # Save ground truth
        gt_info = {
            "pair_id": pair_id,
            "architecture": int(folder_num),
            "center_x": int(final_gt_x),
            "center_y": int(final_gt_y),
            "is_extreme_hard_negative": bool(is_extreme_hard_neg),
            "is_rgb": bool(is_rgb),
            "degradations_applied": len(report),
            "degradation_details": report,
            "script_name": arch["script_name"]
        }
        with open(gt_json, "w") as f:
            json.dump(gt_info, f, indent=2)
        
        manifest_entries.append(gt_info)
        
        if (i + 1) % 10 == 0 or (i + 1) == num_pairs:
            print(f"  Generated [{i+1}/{num_pairs}] {pair_id} | RGB: {is_rgb} | HardNeg: {is_extreme_hard_neg} | Degradations: {len(report)}")
    
    t1 = time.time()
    manifest_path = os.path.join(output_dir, "extreme_dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"total": num_pairs, "pairs": manifest_entries}, f, indent=2)
    
    print("=" * 80)
    print(f"EXTREME TOUGH DATASET GENERATION COMPLETE! ({num_pairs} pairs in {(t1-t0)/60.0:.2f} min)")
    print(f"Output Location: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    import time
    generate_extreme_tough_dataset(num_pairs=100, output_dir="./extreme_tough_dataset_100", master_seed=999)