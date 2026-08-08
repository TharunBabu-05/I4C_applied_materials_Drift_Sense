#!/usr/bin/env python3
"""
Drift-Sense: Evaluation Pipeline  (v2 -- Difficulty Grading)
=============================================================

Runs the inference algorithm on all generated image pairs and produces:
  - Accuracy metrics (% within tolerance, mean/median pixel error)
  - Per-difficulty-tier accuracy (easy / medium / hard)
  - Computation time statistics
  - Success/failure case visualizations with defect annotations
  - Failure root cause analysis

Difficulty is graded by counting near-equal NCC peaks (periodic ambiguity
depth) at the nominal 10x scale. More near-equal peaks = harder case.

Usage:
  python evaluate.py --data_dir ./generated_data --output_dir ./results

Author: Drift-Sense Team
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import our inference and preprocessing modules
from inference import localize, load_grayscale, preprocess, resize_image


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_error(predicted, ground_truth):
    """
    Compute Euclidean pixel error between predicted and ground truth.

    Parameters
    ----------
    predicted : tuple (x, y)
        Predicted center coordinates.
    ground_truth : dict
        Must contain 'center_x' and 'center_y'.

    Returns
    -------
    error : float
        Euclidean distance in pixels.
    """
    px, py = predicted
    gx = ground_truth["center_x"]
    gy = ground_truth["center_y"]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2)


def classify_failure(predicted, ground_truth, search_image, template_size=100):
    """
    Classify the type of failure for root cause analysis.

    Categories:
      - "periodic_ambiguity": landed on wrong grid tile (error is ~N * pitch)
      - "noise_induced": random noise created a false peak
      - "edge_effect": reference was near image boundary
      - "success": within tolerance

    Parameters
    ----------
    predicted : tuple (x, y)
    ground_truth : dict
    search_image : np.ndarray
    template_size : int

    Returns
    -------
    category : str
    explanation : str
    """
    error = compute_error(predicted, ground_truth)

    if error <= 5:
        return "success", "Prediction within tolerance (≤5 pixels)."

    px, py = predicted
    gx = ground_truth["center_x"]
    gy = ground_truth["center_y"]

    h, w = search_image.shape[:2]
    half_t = template_size // 2

    # Check if ground truth is near edge
    if (gx < half_t or gx > w - half_t or
            gy < half_t or gy > h - half_t):
        return "edge_effect", (
            f"Ground truth at ({gx},{gy}) is near image boundary. "
            f"Template extends beyond image edges, degrading NCC reliability."
        )

    # Check if error is roughly a multiple of the expected pitch
    # DRAM pitch at 10x ≈ 2.5-4.5 pixels (25-45 px at 100x, /10)
    dx = abs(px - gx)
    dy = abs(py - gy)

    # If error is between 2-10 pixels in one axis, likely periodic ambiguity
    if (2 < dx < 50 or 2 < dy < 50):
        # Check if the error aligns with pitch multiples
        for pitch_est in range(2, 8):
            if (abs(dx % pitch_est) < 1.5 or abs(dy % pitch_est) < 1.5):
                return "periodic_ambiguity", (
                    f"Error ({dx:.1f}, {dy:.1f}) px aligns with grid pitch "
                    f"(~{pitch_est} px at 10x). The algorithm landed on an "
                    f"adjacent repeating tile in the DRAM array. This is the "
                    f"core challenge: high periodicity creates visually identical "
                    f"match candidates."
                )

    # General noise-induced error
    return "noise_induced", (
        f"Error of {error:.1f} px doesn't align with grid pitch. "
        f"Likely caused by noise-induced false correlation peak. "
        f"The search image's higher noise level created a spurious match."
    )


def grade_difficulty(reference_path, search_path, template_size=100):
    """
    Grade the difficulty of a pair as easy / medium / hard.

    Method: count distinct regional NCC peaks above 80% of the global maximum
    using scipy.ndimage.label (connected components on a binary threshold map).
    More distinct regions => higher periodic ambiguity => harder match.

    Fixed from v2: the old version divided the raw pixel count by template_area//4
    which was too aggressive (2500 for a 100px template), making every pair "easy".
    This version counts actual distinct connected regions using morphological labeling.

    Thresholds (calibrated for DRAM capacitor-body images):
      easy   : 1-5   distinct NCC peaks above 80% threshold
      medium : 6-15  distinct peaks
      hard   : >15   distinct peaks

    Parameters
    ----------
    reference_path : str
    search_path : str
    template_size : int

    Returns
    -------
    difficulty : str  ('easy' | 'medium' | 'hard')
    num_peaks  : int  (number of distinct regional peaks counted)
    """
    try:
        from scipy import ndimage as _ndimage

        ref = load_grayscale(reference_path)
        search = load_grayscale(search_path)
        ref_proc = preprocess(ref, denoise_sigma=0.5)
        search_proc = preprocess(search, denoise_sigma=0.8)
        template = resize_image(ref_proc, (template_size, template_size))

        res = cv2.matchTemplate(
            search_proc.astype(np.float32),
            template.astype(np.float32),
            cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val <= 0:
            return "medium", -1

        # Threshold at 80% of global maximum (generous -- captures all plausible candidates)
        threshold = max_val * 0.80
        binary_map = (res >= threshold).astype(np.uint8)

        # Count distinct connected regions using morphological labeling
        labeled, num_regions = _ndimage.label(binary_map)

        if num_regions <= 5:
            return "easy", num_regions
        elif num_regions <= 15:
            return "medium", num_regions
        else:
            return "hard", num_regions
    except Exception:
        return "medium", -1



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


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def evaluate(data_dir, output_dir, tolerance=5, verbose=True, use_edge=False, use_robust=False):
    """
    Run evaluation on all image pairs in data_dir.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing pair_001/, pair_002/, etc.
    output_dir : str or Path
        Directory to save results.
    tolerance : float
        Pixel tolerance for "success" classification.
    verbose : bool
        Print per-pair results.
    use_edge : bool
        Enable edge enhancement in inference.
    use_robust : bool
        Enable robust preprocessing in inference.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all pair directories
    pair_dirs = sorted([d for d in data_dir.iterdir()
                        if d.is_dir() and d.name.startswith("pair_")])

    if not pair_dirs:
        print(f"ERROR: No pair directories found in {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Drift-Sense: Evaluation Pipeline")
    print("=" * 60)
    print(f"  Data dir:   {data_dir.resolve()}")
    print(f"  Pairs:      {len(pair_dirs)}")
    print(f"  Tolerance:  {tolerance} px")
    print("=" * 60)

    results = []
    errors = []
    times = []
    failures = []
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    difficulty_successes = {"easy": 0, "medium": 0, "hard": 0}

    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        ref_path = pair_dir / "reference.png"
        search_path = pair_dir / "search.png"
        gt_path = pair_dir / "ground_truth.json"

        if not all(p.exists() for p in [ref_path, search_path, gt_path]):
            print(f"  SKIP {pair_name}: missing files")
            continue

        # Load ground truth
        with open(gt_path) as f:
            ground_truth = json.load(f)

        # Run inference
        t_start = time.time()
        predicted = localize(str(ref_path), str(search_path), verbose=False, 
                           use_edge=use_edge, use_robust=use_robust)
        t_elapsed = time.time() - t_start

        # Compute error
        error = compute_error(predicted, ground_truth)
        is_success = bool(error <= tolerance)

        # Classify failure
        search_img = np.array(Image.open(search_path).convert('L'))
        category, explanation = classify_failure(
            predicted, ground_truth, search_img
        )

        # Grade difficulty of this pair
        difficulty, num_peaks = grade_difficulty(str(ref_path), str(search_path))
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        if is_success:
            difficulty_successes[difficulty] = difficulty_successes.get(difficulty, 0) + 1

        result = {
            "pair": pair_name,
            "predicted_x": int(predicted[0]),
            "predicted_y": int(predicted[1]),
            "true_x": int(ground_truth["center_x"]),
            "true_y": int(ground_truth["center_y"]),
            "error_px": round(float(error), 2),
            "success": bool(is_success),
            "category": category,
            "explanation": explanation,
            "time_sec": round(t_elapsed, 3),
            "difficulty": difficulty,
            "num_ambiguous_peaks": num_peaks,
            "defects": ground_truth.get("defects", []),
        }
        results.append(result)
        errors.append(error)
        times.append(t_elapsed)

        if not is_success:
            failures.append(result)

        status = "PASS" if is_success else "FAIL"
        if verbose:
            defect_str = f" [{len(ground_truth.get('defects', []))} defects]" if ground_truth.get('defects') else ""
            print(f"  {status} {pair_name}: pred=({predicted[0]},{predicted[1]}) "
                  f"true=({ground_truth['center_x']},{ground_truth['center_y']}) "
                  f"error={error:.1f}px time={t_elapsed:.2f}s [{category}|{difficulty}]{defect_str}")

    # --- Compute Summary Statistics ---
    errors = np.array(errors)
    times = np.array(times)

    num_total = len(results)
    num_success = sum(1 for r in results if r["success"])
    accuracy = num_success / num_total * 100 if num_total > 0 else 0

    # Per-difficulty accuracy
    diff_accuracy = {}
    for diff in ["easy", "medium", "hard"]:
        total_d = difficulty_counts.get(diff, 0)
        succ_d = difficulty_successes.get(diff, 0)
        diff_accuracy[diff] = {
            "total": total_d,
            "successes": succ_d,
            "accuracy_pct": round(succ_d / total_d * 100, 1) if total_d > 0 else 0.0,
        }

    summary = {
        "total_pairs": num_total,
        "successes": num_success,
        "failures": num_total - num_success,
        "accuracy_pct": round(accuracy, 2),
        "tolerance_px": tolerance,
        "mean_error_px": round(float(errors.mean()), 2),
        "median_error_px": round(float(np.median(errors)), 2),
        "max_error_px": round(float(errors.max()), 2),
        "min_error_px": round(float(errors.min()), 2),
        "std_error_px": round(float(errors.std()), 2),
        "mean_time_sec": round(float(times.mean()), 3),
        "median_time_sec": round(float(np.median(times)), 3),
        "total_time_sec": round(float(times.sum()), 2),
        "difficulty_breakdown": diff_accuracy,
    }

    # --- Save Results ---
    report = {
        "summary": summary,
        "results": results,
    }
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # --- Print Summary ---
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Accuracy:       {accuracy:.1f}% ({num_success}/{num_total}) "
          f"within {tolerance}px tolerance")
    print(f"  Mean error:     {errors.mean():.2f} px")
    print(f"  Median error:   {np.median(errors):.2f} px")
    print(f"  Max error:      {errors.max():.2f} px")
    print(f"  Mean time:      {times.mean():.3f} s per pair")
    print(f"  Total time:     {times.sum():.1f} s")
    print(f"\n  DIFFICULTY BREAKDOWN:")
    for diff in ["easy", "medium", "hard"]:
        d = diff_accuracy[diff]
        if d["total"] > 0:
            print(f"    {diff.capitalize():6s}: {d['accuracy_pct']:.1f}%"
                  f"  ({d['successes']}/{d['total']} pairs)")

    # --- Failure Analysis ---
    if failures:
        print(f"\n  FAILURE ANALYSIS ({len(failures)} failures):")
        # Count failure categories
        categories = {}
        for f_res in failures:
            cat = f_res["category"]
            categories[cat] = categories.get(cat, 0) + 1

        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {count} cases")

        # Show worst failure details
        worst = max(failures, key=lambda x: x["error_px"])
        print(f"\n  WORST FAILURE: {worst['pair']}")
        print(f"    Error: {worst['error_px']} px")
        print(f"    Category: {worst['category']}")
        print(f"    Explanation: {worst['explanation']}")

    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")

    # Success example (lowest error)
    if num_success > 0:
        best_result = min([r for r in results if r["success"]],
                          key=lambda x: x["error_px"])
        best_pair_dir = data_dir / best_result["pair"]
        create_visualization(
            str(best_pair_dir / "reference.png"),
            str(best_pair_dir / "search.png"),
            (best_result["predicted_x"], best_result["predicted_y"]),
            {"center_x": best_result["true_x"],
             "center_y": best_result["true_y"]},
            best_result["error_px"],
            f"SUCCESS: {best_result['pair']}",
            output_dir / "success_example.png"
        )
        print(f"  Saved success_example.png ({best_result['pair']})")

    # Failure example (highest error)
    if failures:
        worst_result = max(failures, key=lambda x: x["error_px"])
        worst_pair_dir = data_dir / worst_result["pair"]
        create_visualization(
            str(worst_pair_dir / "reference.png"),
            str(worst_pair_dir / "search.png"),
            (worst_result["predicted_x"], worst_result["predicted_y"]),
            {"center_x": worst_result["true_x"],
             "center_y": worst_result["true_y"]},
            worst_result["error_px"],
            f"FAILURE: {worst_result['pair']} ({worst_result['category']})",
            output_dir / "failure_example.png"
        )
        print(f"  Saved failure_example.png ({worst_result['pair']})")

    # --- Error Distribution Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Error histogram with difficulty coloring
    diff_colors = {"easy": "#4CAF50", "medium": "#FFC107", "hard": "#F44336"}
    for diff, color in diff_colors.items():
        diff_errors = [r["error_px"] for r in results if r["difficulty"] == diff]
        if diff_errors:
            axes[0].hist(diff_errors, bins=20, color=color, edgecolor='none',
                         alpha=0.75, label=f'{diff.capitalize()} ({len(diff_errors)})')
    axes[0].axvline(tolerance, color='white', linestyle='--', linewidth=2,
                    label=f'Tolerance ({tolerance}px)')
    axes[0].set_xlabel('Pixel Error', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Error Distribution by Difficulty', fontsize=13)
    axes[0].legend(fontsize=9, facecolor='#222', labelcolor='white')

    # Per-difficulty accuracy bar chart
    diffs = ["Easy", "Medium", "Hard"]
    accs = [diff_accuracy[d.lower()]["accuracy_pct"] for d in diffs]
    bars = axes[1].bar(diffs, accs,
                       color=["#4CAF50", "#FFC107", "#F44336"],
                       edgecolor='none', alpha=0.85)
    axes[1].axhline(accuracy, color='white', linestyle='--', linewidth=1.5,
                    label=f'Overall {accuracy:.1f}%')
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{acc:.0f}%', ha='center', va='bottom', color='white', fontsize=11)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy by Difficulty Tier', fontsize=13)
    axes[1].legend(fontsize=9, facecolor='#222', labelcolor='white')

    # Computation time distribution
    axes[2].hist(times, bins=20, color='#7c4dff', edgecolor='none', alpha=0.85)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_title('Computation Time Distribution', fontsize=13)

    plt.tight_layout()
    plt.savefig(str(output_dir / "error_distribution.png"), dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved error_distribution.png")

    print("\n" + "=" * 60)
    print(f"All results saved to: {output_dir.resolve()}")
    print("=" * 60)

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense: Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --data_dir ./generated_data --output_dir ./results
  python evaluate.py --data_dir ./generated_data --output_dir ./results --tolerance 3
        """
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing generated image pairs")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results (default: ./results)")
    parser.add_argument("--tolerance", type=float, default=5,
                        help="Pixel tolerance for success (default: 5)")
    parser.add_argument("--use_edge", action="store_true",
                        help="Enable edge enhancement in inference")
    parser.add_argument("--use_robust", action="store_true",
                        help="Enable robust preprocessing in inference")

    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, tolerance=args.tolerance,
            use_edge=args.use_edge, use_robust=args.use_robust)


if __name__ == "__main__":
    main()
