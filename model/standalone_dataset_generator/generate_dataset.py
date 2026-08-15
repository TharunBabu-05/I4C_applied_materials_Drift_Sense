#!/usr/bin/env python3
"""
Standalone Synthetic SEM Dataset Generator CLI
==============================================

Generates N synthetic SEM image pairs across 60 semiconductor layout architectures
with strict separation:

- 50 Architecture Generators -> TRAIN SET (train/)
- 5 Architecture Generators  -> VALIDATION SET (val/)
- 5 Architecture Generators  -> TEST SET (test/) [HIDDEN & ISOLATED FROM TRAINING]

Includes hard-negative periodic replica shifts and automatic memory management.

Usage:
------
    python generate_dataset.py --num_pairs 10000 --output_dir ./my_dataset_10k
"""

import argparse
import gc
import json
import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from master_generator_v2 import discover_render_functions, generate_clean_pair
from degradation_engine import apply_full_degradation, DEFAULT_CONFIG


def generate_dataset(num_pairs=10000, output_dir="./dataset", master_seed=42):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_60_scripts")
    registry = discover_render_functions(scripts_dir)
    num_arch = len(registry)

    if num_arch == 0:
        raise RuntimeError(f"No generator scripts found in {scripts_dir}")

    # Deterministic partition: 50 Train / 5 Val / 5 Test generators (Hidden Test Set)
    master_rng = np.random.default_rng(master_seed)
    indices = np.arange(num_arch)
    master_rng.shuffle(indices)

    train_arch_indices = indices[:50]
    val_arch_indices = indices[50:55]
    test_arch_indices = indices[55:]

    train_count = int(round(num_pairs * 0.80))
    val_count = int(round(num_pairs * 0.10))
    test_count = num_pairs - train_count - val_count

    print("=" * 85)
    print(f"STANDALONE SEM DATASET GENERATOR ({num_pairs:,} PAIRS)")
    print("=" * 85)
    print(f"Output Directory:    {output_dir}")
    print(f"Total Generators:    {num_arch}")
    print(f"Train Generators:    50 scripts ({train_count:,} pairs -> train/)")
    print(f"Val Generators:      5 scripts ({val_count:,} pairs -> val/)")
    print(f"Test Generators:     5 HIDDEN scripts ({test_count:,} pairs -> test/) [ISOLATED]")
    print("=" * 85)

    splits = [
        ("train", train_count, train_arch_indices),
        ("val", val_count, val_arch_indices),
        ("test", test_count, test_arch_indices)
    ]

    manifest_entries = []
    config = DEFAULT_CONFIG.copy()
    t0 = time.time()

    total_generated = 0
    for split_name, count_needed, arch_indices in splits:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for i in range(count_needed):
            pair_id = f"pair_{i+1:04d}" if count_needed >= 1000 else f"pair_{i+1:03d}"
            pair_dir = os.path.join(split_dir, pair_id)
            os.makedirs(pair_dir, exist_ok=True)

            gt_json = os.path.join(pair_dir, "groundtruth.json")
            search_png = os.path.join(pair_dir, "search.png")
            target_png = os.path.join(pair_dir, "target.png")

            # Resumable generation check
            if os.path.exists(gt_json) and os.path.exists(search_png) and os.path.exists(target_png):
                with open(gt_json, "r") as f:
                    gt_info = json.load(f)
                manifest_entries.append(gt_info)
                total_generated += 1
                if total_generated % 500 == 0:
                    print(f"  [Existing] [{total_generated}/{num_pairs}] pairs | Split: {split_name:<5}")
                continue

            arch_idx = arch_indices[i % len(arch_indices)]
            arch = registry[arch_idx]
            folder_num = arch["folder_num"]
            render_fn = arch["render_fn"]
            generate_fn = arch["generate_fn"]

            render_seed = int(master_rng.integers(0, 2**31))
            degrade_seed = int(master_rng.integers(0, 2**31))
            degrade_rng = np.random.default_rng(degrade_seed)

            # Hard-Negative periodic pitch shift injection (25% of cases)
            is_periodic_hard_neg = (i % 4 == 0)
            
            # Generate random base coordinates across the ENTIRE valid image area (avoiding extreme 50px edges)
            # This ensures targets can be in top-left, bottom-right, etc.
            base_x = int(degrade_rng.integers(100, 900))
            base_y = int(degrade_rng.integers(100, 900))
            
            if is_periodic_hard_neg:
                # Add the hard negative shift to the random base coordinate
                dx_shift = int(degrade_rng.choice([30, -30, 45, -45, 60, -60]))
                dy_shift = int(degrade_rng.choice([43, -43, 30, -30, 52, -52]))
                gt_x = max(100, min(900, base_x + dx_shift))
                gt_y = max(100, min(900, base_y + dy_shift))
            else:
                gt_x, gt_y = base_x, base_y

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

            cv2.imwrite(os.path.join(pair_dir, "target.png"), target_clean)
            cv2.imwrite(os.path.join(pair_dir, "reference.png"), target_clean)

            search_degraded, final_gt_x, final_gt_y, report = apply_full_degradation(
                search_clean, gt_x, gt_y, degrade_rng, config
            )
            cv2.imwrite(os.path.join(pair_dir, "search.png"), search_degraded)

            del search_clean, target_clean, search_degraded
            if total_generated % 50 == 0:
                gc.collect()

            gt_info = {
                "pair_id": pair_id,
                "split": split_name,
                "architecture": folder_num,
                "center_x": int(final_gt_x),
                "center_y": int(final_gt_y),
                "is_hard_negative": is_periodic_hard_neg,
                "script_name": arch["script_name"]
            }
            with open(os.path.join(pair_dir, "groundtruth.json"), "w") as f:
                json.dump(gt_info, f, indent=2)

            manifest_entries.append(gt_info)
            total_generated += 1

            if total_generated % 500 == 0 or total_generated == num_pairs:
                print(f"  Generated [{total_generated}/{num_pairs}] pairs | Split: {split_name:<5} | Arch {folder_num}")

    t1 = time.time()
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"total": total_generated, "pairs": manifest_entries}, f, indent=2)

    print("=" * 85)
    print(f"DATASET GENERATION COMPLETE! ({total_generated:,} pairs in {(t1-t0)/60.0:.2f} min)")
    print(f"Output Location: {output_dir}")
    print(f"Manifest:        {manifest_path}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Standalone SEM Dataset Generator")
    parser.add_argument("--num_pairs", type=int, default=10000, help="Total pairs to generate")
    parser.add_argument("--output_dir", type=str, default="./dataset_10k", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")

    args = parser.parse_args()
    generate_dataset(num_pairs=args.num_pairs, output_dir=args.output_dir, master_seed=args.seed)


if __name__ == "__main__":
    main()
