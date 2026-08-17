#!/usr/bin/env python3
"""
Multi-Landmark Dataset Generation & Nearest-to-Center Inference Evaluation
==========================================================================

1. Generates dataset pairs containing 2 or 3 repeating black mark landmarks.
2. Identifies the ground truth landmark nearest to search image center (500, 500).
3. Runs localization inference and evaluates disambiguation accuracy.
4. Draws bounding boxes for all candidate landmarks (blue) and highlights predicted match (green) vs ground truth (red).
5. Saves visualization images to ./multi_landmark_bbox_results/
"""

import os
import gc
import json
import time
import cv2
import numpy as np
from PIL import Image

from dataset_generator import generate_image_pair
from inference import localize

def run_bbox_evaluation():
    output_data_dir = "./multi_landmark_dataset"
    results_dir = "./multi_landmark_bbox_results"
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    num_pairs = 5
    seed = 100
    summary_records = []

    print("=" * 80)
    print("EVALUATING MULTI-LANDMARK DATASET (2-3 Black Marks, Nearest to Center Target)")
    print("=" * 80)

    for i in range(1, num_pairs + 1):
        gc.collect()
        print(f"\n[Pair {i:03d}] Generating layout with 2-3 black mark landmarks...")
        rng_i = np.random.default_rng(seed + i * 13)
        ref_img, search_img, gt, meta = generate_image_pair(pair_index=i, style="RING", rng=rng_i)

        pair_dir = os.path.join(output_data_dir, f"pair_{i:03d}")
        os.makedirs(pair_dir, exist_ok=True)

        ref_path = os.path.join(pair_dir, "reference.png")
        search_path = os.path.join(pair_dir, "search.png")
        gt_path = os.path.join(pair_dir, "ground_truth.json")

        Image.fromarray(ref_img, mode='L').save(ref_path)
        Image.fromarray(search_img, mode='L').save(search_path)
        with open(gt_path, 'w') as f:
            json.dump(gt, f, indent=2)

        gt_x, gt_y = gt["center_x"], gt["center_y"]
        defects = gt.get("defects", [{}])[0]
        landmarks_master = defects.get("landmarks", [])

        print(f"  Landmarks generated (master scale): {landmarks_master}")
        print(f"  Target Nearest to Center (GT)     : ({gt_x}, {gt_y})")

        # Run Inference
        print("  Running Localization Inference...")
        t0 = time.time()
        pred_x, pred_y = localize(ref_path, search_path, verbose=False)
        dt = time.time() - t0

        err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        status = "SUCCESS" if err <= 5.0 else "FAIL"

        print(f"  Inference Result                 : ({pred_x}, {pred_y}) | Error: {err:.2f} px | Time: {dt:.3f}s | Status: {status}")

        # Draw Bounding Box Visualizations
        search_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        box_w, box_h = 100, 100

        # Draw all landmarks in Search Image space in BLUE/CYAN
        for idx, (lm_mx, lm_my) in enumerate(landmarks_master, start=1):
            sx, sy = int(round(lm_mx / 10.0)), int(round(lm_my / 10.0))
            if 0 <= sx <= 1000 and 0 <= sy <= 1000:
                cv2.rectangle(search_color, (sx - 45, sy - 45), (sx + 45, sy + 45), (255, 200, 0), 1)
                cv2.putText(search_color, f"Landmark #{idx}", (sx - 40, max(12, sy - 50)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA)

        # Ground Truth Target Nearest to Center (RED Box)
        gt_x1, gt_y1 = max(0, gt_x - box_w // 2), max(0, gt_y - box_h // 2)
        gt_x2, gt_y2 = min(1000, gt_x + box_w // 2), min(1000, gt_y + box_h // 2)
        cv2.rectangle(search_color, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 0, 255), 2)
        cv2.circle(search_color, (gt_x, gt_y), 4, (0, 0, 255), -1)
        cv2.putText(search_color, f"GT (Nearest Center) ({gt_x},{gt_y})", (gt_x1, max(15, gt_y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Predicted Match (GREEN Box)
        pred_x1, pred_y1 = max(0, pred_x - box_w // 2), max(0, pred_y - box_h // 2)
        pred_x2, pred_y2 = min(1000, pred_x + box_w // 2), min(1000, pred_y + box_h // 2)
        cv2.rectangle(search_color, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 2)
        cv2.circle(search_color, (pred_x, pred_y), 3, (0, 255, 0), -1)
        cv2.putText(search_color, f"Pred Match ({pred_x},{pred_y})", (pred_x1, min(990, pred_y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Header Info Overlay
        info_text = f"Pair {i:03d} | Landmarks: {len(landmarks_master)} | Err: {err:.2f}px | Time: {dt:.3f}s"
        cv2.rectangle(search_color, (0, 0), (1000, 30), (0, 0, 0), -1)
        cv2.putText(search_color, info_text, (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Reference Image on left + Search Image with BBoxes on right
        ref_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(ref_color, (0, 0), (1000, 30), (0, 0, 0), -1)
        cv2.putText(ref_color, f"Reference Target (Pair {i:03d})", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        composite = np.hstack([ref_color, search_color])

        save_name = f"result_pair_{i:03d}_multilandmark_bbox.png"
        save_path = os.path.join(results_dir, save_name)
        cv2.imwrite(save_path, composite)
        print(f"  Saved Bounding Box Result -> {save_path}")

        summary_records.append({
            "pair": i,
            "num_landmarks": len(landmarks_master),
            "gt": (gt_x, gt_y),
            "pred": (pred_x, pred_y),
            "error_px": err,
            "time_sec": dt,
            "status": status,
            "result_image": save_path
        })

    # Print Summary Table
    print("\n" + "=" * 85)
    print("SUMMARY RESULTS: MULTI-LANDMARK REPEATING BLACK MARKS (NEAREST TO CENTER TARGET)")
    print("=" * 85)
    print(f"{'Pair':<6} | {'Landmarks':<10} | {'GT Nearest Center':<18} | {'Predicted Center':<18} | {'Error (px)':<10} | {'Status'}")
    print("-" * 85)
    for r in summary_records:
        gt_s = f"({r['gt'][0]},{r['gt'][1]})"
        pr_s = f"({r['pred'][0]},{r['pred'][1]})"
        print(f"{r['pair']:<6} | {r['num_landmarks']:<10} | {gt_s:<18} | {pr_s:<18} | {r['error_px']:<10.2f} | {r['status']}")
    print("=" * 85)

if __name__ == "__main__":
    run_bbox_evaluation()
