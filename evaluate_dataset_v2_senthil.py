#!/usr/bin/env python3
"""
Evaluate Senthil's fast inference on dataset_v2_two_noise_100_14-8-26
"""

import json
import os
import time
from pathlib import Path
import numpy as np

# Import Senthil's fast inference
import fast_inference_senthil as senthil_localize

def compute_error(predicted, ground_truth):
    """Compute Euclidean pixel error between predicted and ground truth."""
    px, py = predicted
    gx = ground_truth["center_x"]
    gy = ground_truth["center_y"]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

def main():
    data_dir = r"C:\Semester-7\I4C_hackathon\dataset_v2_two_noise_100_14-8-26"
    results_dir = Path(data_dir) / "results_senthil_fast"
    results_dir.mkdir(exist_ok=True)
    
    # Get all pair directories
    data_path = Path(data_dir)
    pair_dirs = sorted([d for d in data_path.iterdir()
                        if d.is_dir() and d.name.startswith("pair_")])
    
    print("=" * 80)
    print("EVALUATING SENTHIL'S FAST INFERENCE ON DATASET_V2_TWO_NOISE_100")
    print("=" * 80)
    print(f"Data dir: {data_dir}")
    print(f"Pairs: {len(pair_dirs)}")
    print(f"Tolerance: 5 px")
    print("=" * 80)
    
    results = []
    errors = []
    times = []
    
    for pair_dir in pair_dirs:
        pair_name = pair_dir.name
        # Try reference.png first, if not found try target.png
        ref_path = pair_dir / "reference.png"
        if not ref_path.exists():
            ref_path = pair_dir / "target.png"
        search_path = pair_dir / "search.png"
        gt_path = pair_dir / "groundtruth.json"
        
        if not all(p.exists() for p in [ref_path, search_path, gt_path]):
            print(f"  SKIP {pair_name}: missing files")
            continue
        
        # Load ground truth
        with open(gt_path) as f:
            ground_truth = json.load(f)
        
        # Run inference
        t_start = time.perf_counter()
        try:
            predicted = senthil_localize.localize(str(ref_path), str(search_path), verbose=False)
            t_elapsed = time.perf_counter() - t_start
            
            # Compute error
            error = compute_error(predicted, ground_truth)
            is_success = bool(error <= 5)
            
            status = "PASS" if is_success else "FAIL"
            print(f"  {status} {pair_name}: pred={predicted} true=({ground_truth['center_x']},{ground_truth['center_y']}) error={error:.1f}px time={t_elapsed*1000:.1f}ms")
            
            results.append({
                "pair": pair_name,
                "predicted_x": predicted[0],
                "predicted_y": predicted[1],
                "true_x": ground_truth["center_x"],
                "true_y": ground_truth["center_y"],
                "error_px": error,
                "success": is_success,
                "time_sec": t_elapsed
            })
            
            errors.append(error)
            times.append(t_elapsed)
            
        except Exception as e:
            print(f"  ERROR {pair_name}: {e}")
            results.append({
                "pair": pair_name,
                "error": str(e),
                "success": False,
                "time_sec": 0
            })
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    num_success = sum(1 for r in results if r["success"])
    num_total = len(results)
    accuracy = (num_success / num_total * 100) if num_total > 0 else 0
    
    if errors:
        errors_array = np.array(errors)
        times_array = np.array(times)
        
        print(f"  Accuracy:       {accuracy:.1f}% ({num_success}/{num_total}) within 5px tolerance")
        print(f"  Mean error:     {errors_array.mean():.2f} px")
        print(f"  Median error:   {np.median(errors_array):.2f} px")
        print(f"  Max error:      {errors_array.max():.2f} px")
        print(f"  Min error:      {errors_array.min():.2f} px")
        print(f"  Std error:      {errors_array.std():.2f} px")
        print(f"  Mean time:      {times_array.mean()*1000:.2f} ms per pair")
        print(f"  Median time:    {np.median(times_array)*1000:.2f} ms per pair")
        print(f"  Total time:     {times_array.sum():.2f} s")
    else:
        print("  No valid results")
    
    # Save results
    report = {
        "summary": {
            "total_pairs": num_total,
            "successes": num_success,
            "failures": num_total - num_success,
            "accuracy_pct": round(accuracy, 2),
            "tolerance_px": 5,
            "mean_error_px": round(float(np.mean(errors)), 2) if errors else 0,
            "median_error_px": round(float(np.median(errors)), 2) if errors else 0,
            "max_error_px": round(float(np.max(errors)), 2) if errors else 0,
            "min_error_px": round(float(np.min(errors)), 2) if errors else 0,
            "std_error_px": round(float(np.std(errors)), 2) if errors else 0,
            "mean_time_ms": round(float(np.mean(times) * 1000), 2) if times else 0,
            "median_time_ms": round(float(np.median(times) * 1000), 2) if times else 0,
            "total_time_sec": round(float(np.sum(times)), 2) if times else 0
        },
        "results": results
    }
    
    report_path = results_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAll results saved to: {results_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()