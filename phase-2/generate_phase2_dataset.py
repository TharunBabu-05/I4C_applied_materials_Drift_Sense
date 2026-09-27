#!/usr/bin/env python3
"""
Phase-2 Fixed Dataset Generator — 16,000 Training Pairs
=========================================================
Correctly simulates real SEM camera misalignment by applying
global scale and rotation transforms to the ENTIRE search image.

Key fixes over the broken generator:
  1. Global affine transform (scale + rotation) applied to the WHOLE image
     — no more patch blending that creates unscaled background decoys.
  2. Ground truth coordinates are correctly mapped through the affine matrix.
  3. Reference crop is taken BEFORE transforming (clean, unblurred patch).
  4. Full Phase-2 requirements:
       Set A  (45%): Nominal — scale∈[8,12], theta∈[-5,5], no degradation
       Set B  (35%): Degraded — same pose, 4-level SEM noise (blur, charge, jitter, Poisson)
       Set C  (15%): Absent — different die region, found=0
       Set D  (5%):  Optical — RGB 3-channel analogue, reference present
  5. Subpixel-accurate ground truth.
  6. Resumable — skips already-generated pairs.

Usage:
    python generate_phase2_dataset.py --num_pairs 16000 --output_dir ./phase2_training_16k
"""

import os
import sys
import json
import math
import random
import csv
import gc
import argparse
import time
import cv2
import numpy as np
from scipy import ndimage

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN60_DIR = os.path.join(BASE_DIR, "phase2_generator_60")
SCRIPTS_DIR = os.path.join(GEN60_DIR, "clean_60_scripts")
sys.path.insert(0, GEN60_DIR)
from master_generator_v2 import discover_render_functions

# ── Default GT map for the 60 generator scripts ───────────────────────────────
DEFAULT_GT_MAP = {
    "001": (650, 350), "002": (650, 350), "003": (490, 490), "004": (350, 630),
    "005": (500, 500), "006": (300, 700), "007": (490, 490), "008": (350, 630),
    "009": (460, 460), "010": (420, 580), "011": (500, 500), "012": (500, 335),
    "013": (500, 690), "014": (310, 500), "015": (500, 405), "016": (310, 500),
    "017": (500, 310), "018": (500, 500), "019": (488, 506), "020": (488, 314),
    "021": (500, 500), "022": (500, 310), "023": (500, 500), "024": (500, 310),
    "025": (500, 500), "026": (500, 310), "027": (500, 500), "028": (500, 310),
    "029": (500, 500), "030": (500, 310), "031": (500, 500), "032": (500, 310),
    "033": (500, 500), "034": (500, 310), "035": (500, 500), "036": (500, 310),
    "037": (500, 500), "038": (500, 310), "039": (500, 500), "040": (500, 310),
    "041": (500, 500), "042": (500, 310), "043": (500, 500), "044": (500, 310),
    "045": (500, 500), "046": (500, 310), "047": (500, 500), "048": (500, 310),
    "049": (500, 500), "050": (500, 310), "051": (500, 500), "052": (500, 310),
    "053": (500, 500), "054": (500, 310), "055": (500, 500), "056": (500, 310),
    "057": (500, 500), "058": (500, 310), "059": (500, 500), "060": (500, 310),
}


def apply_set_b_degradation(img_uint8: np.ndarray, severity: int, rng) -> np.ndarray:
    """
    Apply SEM-inspired degradation at one of 4 severity levels.
    Models: Gaussian blur, charge drift gradient, scan-line jitter, Poisson + Gaussian noise.
    """
    if len(img_uint8.shape) == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    img = img_uint8.astype(np.float32)
    h, w = img.shape

    # 1. Slight per-image polygon-scale distortion (±20%)
    cd_scale = 1.0 + float(rng.uniform(-0.20, 0.20))
    if abs(cd_scale - 1.0) > 0.05:
        new_w = max(100, int(round(w * cd_scale)))
        new_h = max(100, int(round(h * cd_scale)))
        img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if cd_scale > 1.0:
            y0 = (new_h - h) // 2
            x0 = (new_w - w) // 2
            img = img_r[y0:y0+h, x0:x0+w]
        else:
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            img = np.pad(img_r, ((pad_y, h - new_h - pad_y), (pad_x, w - new_w - pad_x)), mode='edge')

    # 2. Gaussian blur (defocus)
    blur_sigmas = [0.8, 1.5, 2.5, 3.8]
    b_sigma = blur_sigmas[severity - 1] + float(rng.uniform(-0.2, 0.2))
    if b_sigma > 0.3:
        img = ndimage.gaussian_filter(img, sigma=b_sigma)

    # 3. Charge drift gradient
    charge_amp = [10.0, 25.0, 45.0, 70.0][severity - 1]
    y_grad = np.linspace(-1, 1, h, dtype=np.float32)[:, np.newaxis]
    x_grad = np.linspace(-1, 1, w, dtype=np.float32)[np.newaxis, :]
    img += charge_amp * (0.6 * y_grad + 0.4 * x_grad)

    # 4. Scan-line jitter
    jitter_amps = [0.5, 1.5, 3.5, 6.0]
    j_amp = jitter_amps[severity - 1]
    if j_amp > 0:
        freq = float(rng.uniform(0.01, 0.05))
        phase = float(rng.uniform(0, 2 * np.pi))
        shifts = j_amp * np.sin(2 * np.pi * freq * np.arange(h) + phase)
        for r in range(h):
            s = int(round(shifts[r]))
            if s != 0:
                img[r] = np.roll(img[r], s)

    # 5. Poisson shot noise
    poisson_scale = [12.0, 8.0, 5.0, 2.5][severity - 1]
    lam = np.clip(img, 0.001, 255) * poisson_scale
    img = rng.poisson(lam).astype(np.float32) / poisson_scale

    # 6. Gaussian readout noise
    gauss_std = [2.0, 4.5, 8.0, 14.0][severity - 1]
    img += rng.normal(0, gauss_std, size=img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)


def convert_to_rgb_optical(img_gray: np.ndarray, rng) -> np.ndarray:
    """Simulate a 3-channel optical microscope analogue of a grayscale SEM image."""
    if len(img_gray.shape) == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    img = img_gray.astype(np.float32) / 255.0
    r = np.clip(img * 220 + 20 + rng.normal(0, 5, img.shape), 0, 255)
    g = np.clip(img * 190 + 40 + rng.normal(0, 5, img.shape), 0, 255)
    b = np.clip(img * 140 + 70 + rng.normal(0, 5, img.shape), 0, 255)
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def render_image(arch_info, gt_x=500, gt_y=500, seed=42,
                 h=1000, w=1000) -> np.ndarray:
    """Try to render a 1000×1000 grayscale image from the generator script."""
    fn = arch_info.get("render_fn") or arch_info.get("generate_fn")
    img = None
    for call in [
        lambda: fn(h=h, w=w, gt_x=int(gt_x), gt_y=int(gt_y), seed=seed),
        lambda: fn(h=h, w=w, seed=seed),
        lambda: fn(),
    ]:
        try:
            res = call()
            img = res[0] if isinstance(res, tuple) else res
            break
        except Exception:
            continue

    if img is None or not isinstance(img, np.ndarray):
        np.random.seed(seed)
        img = np.full((h, w), 50, dtype=np.uint8)
        for yy in range(40, h, 60):
            for xx in range(40, w, 60):
                cv2.rectangle(img, (xx-15, yy-15), (xx+15, yy+15), 180, -1)

    if isinstance(img, str) and os.path.exists(img):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def generate_one_pair(pair_id, pair_dir, arch_info, absent_arch_info,
                      scale, theta, gt_x, gt_y,
                      is_present, is_degraded, is_rgb,
                      severity, pair_seed, pair_rng):
    """
    Generate a single (reference, search, groundtruth) triple.

    CRITICAL FIX: The search image is generated by applying global affine
    (scale + rotation) to the ENTIRE clean image, and the GT coordinates are
    mapped through the same affine matrix. This means there are NO unscaled
    decoys in the background — the transform is physically accurate.
    """
    os.makedirs(pair_dir, exist_ok=True)

    if is_present:
        # ── 1. Render the clean scene at the native GT location ───────────────
        search_clean = render_image(arch_info, gt_x=gt_x, gt_y=gt_y, seed=pair_seed)

        # ── 2. Crop the 100×100 reference patch BEFORE any transform ─────────
        crop = 100
        x0 = max(0, min(1000 - crop, int(round(gt_x)) - crop // 2))
        y0 = max(0, min(1000 - crop, int(round(gt_y)) - crop // 2))
        ref_patch = search_clean[y0:y0+crop, x0:x0+crop].copy()

        # ── 3. Apply global affine (scale+rotation) to the ENTIRE search image
        #      scale/10 = the cv2 scale factor relative to nominal scale=10
        M = cv2.getRotationMatrix2D((500.0, 500.0), theta, scale / 10.0)
        search_transformed = cv2.warpAffine(
            search_clean, M, (1000, 1000),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        # ── 4. Map GT through the same affine ─────────────────────────────────
        pt = np.array([[[gt_x, gt_y]]], dtype=np.float32)
        gt_transformed = cv2.transform(pt, M)[0][0]
        final_gt_x, final_gt_y = float(gt_transformed[0]), float(gt_transformed[1])

        # Clamp — keep inside image canvas (rare edge cases)
        final_gt_x = max(50.0, min(950.0, final_gt_x))
        final_gt_y = max(50.0, min(950.0, final_gt_y))

        # ── 5. Apply SEM degradation to the search image (Set B only) ─────────
        if is_degraded:
            search_out = apply_set_b_degradation(search_transformed, severity, pair_rng)
        else:
            search_out = np.clip(search_transformed, 0, 255).astype(np.uint8)

        ref_out = ref_patch  # clean reference crop, no degradation

    else:
        # ── Absent: use a DIFFERENT die region as reference ───────────────────
        search_clean = render_image(arch_info, seed=pair_seed)
        ref_bg = render_image(absent_arch_info, seed=pair_seed + 99)
        ref_out = ref_bg[450:550, 450:550].copy()
        search_out = search_clean.copy()
        final_gt_x, final_gt_y = 0.0, 0.0

    # ── 6. Convert to RGB for Set D ───────────────────────────────────────────
    if is_rgb:
        ref_save = convert_to_rgb_optical(ref_out, pair_rng)
        search_save = convert_to_rgb_optical(search_out, pair_rng)
    else:
        ref_save = ref_out.copy()
        search_save = search_out.copy()

    # ── 7. Save images and ground truth ───────────────────────────────────────
    cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_save)
    cv2.imwrite(os.path.join(pair_dir, "search.png"), search_save)

    gt_data = {
        "pair_id":  pair_id,
        "target_x": round(final_gt_x, 3),
        "target_y": round(final_gt_y, 3),
        "theta":    round(theta, 4),
        "scale":    round(scale, 4),
        "found":    1 if is_present else 0,
        "set":      "Set A" if (is_present and not is_degraded and not is_rgb)
                    else "Set B" if (is_present and is_degraded)
                    else "Set C" if not is_present
                    else "Set D",
        "severity": severity,
        "seed":     pair_seed,
    }
    with open(os.path.join(pair_dir, "groundtruth.json"), "w") as f:
        json.dump(gt_data, f, indent=2)

    return gt_data


def generate_dataset(num_pairs: int, output_dir: str, seed: int = 2026):
    """
    Generate `num_pairs` Phase-2 training pairs.

    Distribution (matches hackathon spec proportions):
        Set A  45% — present, nominal (no degradation)
        Set B  35% — present, degraded
        Set C  15% — absent
        Set D   5% — present, RGB optical

    Each pair uses one of the 60 generator scripts in equal rotation.
    """
    os.makedirs(output_dir, exist_ok=True)

    registry = discover_render_functions(SCRIPTS_DIR)
    num_arch = len(registry)
    if num_arch == 0:
        raise RuntimeError(f"No generator scripts found in {SCRIPTS_DIR}")
    print(f"Found {num_arch} generator architectures.")

    rng_master = np.random.default_rng(seed)
    random.seed(seed)

    # Build pair-type sequence matching the distribution
    n_a = int(round(num_pairs * 0.45))
    n_b = int(round(num_pairs * 0.35))
    n_c = int(round(num_pairs * 0.15))
    n_d = num_pairs - n_a - n_b - n_c  # remainder → Set D
    pair_types = (
        [("A", False, True,  False)] * n_a +
        [("B", True,  True,  False)] * n_b +
        [("C", False, False, False)] * n_c +
        [("D", False, True,  True)]  * n_d
    )
    random.shuffle(pair_types)

    manifest_rows = []
    skipped = 0
    generated = 0
    t_start = time.time()

    manifest_csv = os.path.join(output_dir, "manifest.csv")
    csv_exists = os.path.exists(manifest_csv)
    csv_file = open(manifest_csv, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "pair_id", "set", "generator_id", "target_x", "target_y",
            "theta", "scale", "found", "severity", "seed"
        ])

    for idx, (set_code, is_degraded, is_present, is_rgb) in enumerate(pair_types):
        pair_num = idx + 1
        pair_id  = f"pair_{pair_num:05d}"
        pair_dir = os.path.join(output_dir, pair_id)
        gt_path  = os.path.join(pair_dir, "groundtruth.json")

        # ── Resumable: skip already-done pairs ────────────────────────────────
        if os.path.exists(gt_path):
            skipped += 1
            continue

        # ── Pick generator architecture ────────────────────────────────────────
        arch_info = registry[(pair_num - 1) % num_arch]
        folder_num = arch_info["folder_num"]
        pair_seed = int(seed + pair_num * 1009)
        pair_rng  = np.random.default_rng(pair_seed)

        # ── Pose parameters ────────────────────────────────────────────────────
        if is_present:
            scale = float(pair_rng.uniform(8.0, 12.0))
            theta = float(pair_rng.uniform(-5.0, 5.0))
            native_x, native_y = DEFAULT_GT_MAP.get(folder_num, (500, 500))
            gt_x = float(native_x + pair_rng.integers(-40, 40))
            gt_y = float(native_y + pair_rng.integers(-40, 40))
            gt_x = max(200.0, min(800.0, gt_x))
            gt_y = max(200.0, min(800.0, gt_y))
        else:
            scale = 0.0
            theta = 0.0
            gt_x  = 0.0
            gt_y  = 0.0

        severity = int(pair_rng.integers(1, 5)) if is_degraded else 0
        absent_arch = registry[(pair_num + 17) % num_arch]

        # ── Generate ──────────────────────────────────────────────────────────
        try:
            gt_data = generate_one_pair(
                pair_id=pair_id,
                pair_dir=pair_dir,
                arch_info=arch_info,
                absent_arch_info=absent_arch,
                scale=scale, theta=theta,
                gt_x=gt_x, gt_y=gt_y,
                is_present=is_present,
                is_degraded=is_degraded,
                is_rgb=is_rgb,
                severity=severity,
                pair_seed=pair_seed,
                pair_rng=pair_rng,
            )

            csv_writer.writerow([
                pair_id, gt_data["set"], f"gen_{folder_num}",
                gt_data["target_x"], gt_data["target_y"],
                gt_data["theta"], gt_data["scale"],
                gt_data["found"], gt_data["severity"], pair_seed
            ])
            csv_file.flush()
            generated += 1

        except Exception as e:
            print(f"  [WARN] {pair_id} failed: {e}")
            continue

        # ── Progress report ────────────────────────────────────────────────────
        if generated % 100 == 0 or pair_num == num_pairs:
            elapsed = time.time() - t_start
            rate = generated / elapsed if elapsed > 0 else 0
            eta  = (num_pairs - pair_num) / rate if rate > 0 else 0
            print(f"  [{pair_num}/{num_pairs}] Generated: {generated}  "
                  f"Skipped: {skipped}  "
                  f"Speed: {rate:.1f} pairs/s  ETA: {eta:.0f}s")

        gc.collect()

    csv_file.close()
    print(f"\n✅ Done! Generated {generated} new pairs + {skipped} already existed.")
    print(f"   Output: {output_dir}")
    print(f"   Manifest: {manifest_csv}")


def main():
    parser = argparse.ArgumentParser(description="Phase-2 Dataset Generator")
    parser.add_argument("--num_pairs", type=int, default=16000,
                        help="Total number of pairs to generate (default: 16000)")
    parser.add_argument("--output_dir", type=str,
                        default="phase2_training_16k",
                        help="Output directory for the dataset")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Master random seed")
    args = parser.parse_args()

    print(f"Phase-2 Fixed Dataset Generator")
    print(f"  Pairs      : {args.num_pairs}")
    print(f"  Output     : {args.output_dir}")
    print(f"  Seed       : {args.seed}")
    print(f"  Set dist.  : A=45% B=35% C=15% D=5%")
    print()

    generate_dataset(args.num_pairs, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
