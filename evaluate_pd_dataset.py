#!/usr/bin/env python3
"""
Evaluate both inference methods on PD dataset
Dataset structure: pd/sample_0001/sample_000X/ with reference_100x.png, search_10x.png
Ground truth format: Uses "search_center_xy" instead of "center_x"/"center_y"
"""

import json
import os
import time
from pathlib import Path
import numpy as np

# Import both inference methods
from inference import localize as your_localize
import fast_inference_senthil as senthil_localize

def compute_error(predicted, ground_truth):
    """Compute Euclidean pixel error between predicted and ground truth."""
    px, py = predicted
    gx, gy = ground_truth["target"]["search_center_xy"]
    return np.sqrt((px - gx) ** 2 + (py - gy) ** 2)

def find_pairs_in_pd_dataset(data_dir):
    """Find all pairs in PD dataset structure"""
    pairs = []
    data_path = Path(data_dir)
    
    # Search recursively for all ground truth files
    for gt_file in data_path.rglob('ground_truth.json'):
        pair_folder = gt_file.parent
        ref_file = pair_folder / 'reference_100x.png'
        search_file = pair_folder / 'search_10x.png'
        
        if ref_file.exists() and search_file.exists():
            # Create a relative name for the pair
            rel_path = pair_folder.relative_to(data_path)
            pairs.append({
                'name': str(rel_path),
                'gt_path': gt_file,
                'ref_path': ref_file,
                'search_path': search_file
            })
    
    return sorted(pairs, key=lambda x: x['name'])

def evaluate_inference_method(pairs, inference_func, method_name):
    """Evaluate a specific inference method on all pairs"""
    print(f"\n{'='*80}")
    print(f"EVALUATING {method_name.upper()} ON PD DATASET")
    print(f"{'='*80}")
    
    results = []
    errors = []
    times = []
    
    for pair_info in pairs:
        pair_name = pair_info['name']
        gt_path = pair_info['gt_path']
        ref_path = pair_info['ref_path']
        search_path = pair_info['search_path']
        
        # Load ground truth
        with open(gt_path) as f:
            ground_truth = json.load(f)
        
        # Run inference
        t_start = time.perf_counter()
        try:
            # Handle different function signatures
            if method_name == "Senthil's Fast Inference":
                predicted = inference_func(str(ref_path), str(search_path), verbose=False)
            else:
                predicted = inference_func(str(ref_path), str(search_path), verbose=False)
            t_elapsed = time.perf_counter() - t_start
            
            # Compute error
            error = compute_error(predicted, ground_truth)
            is_success = bool(error <= 5)
            
            status = "PASS" if is_success else "FAIL"
            gt_center = ground_truth["target"]["search_center_xy"]
            print(f"  {status} {pair_name}: pred={predicted} true={gt_center} error={error:.1f}px time={t_elapsed*1000:.1f}ms")
            
            results.append({
                "pair": pair_name,
                "predicted_x": predicted[0],
                "predicted_y": predicted[1],
                "true_x": gt_center[0],
                "true_y": gt_center[1],
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
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    if errors:
        num_success = sum(1 for r in results if r["success"])
        num_total = len(results)
        accuracy = (num_success / num_total * 100) if num_total > 0 else 0
        
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
    
    return results

def main():
    data_dir = r"C:\Semester-7\I4C_hackathon\pd"
    
    # Find all pairs
    pairs = find_pairs_in_pd_dataset(data_dir)
    print(f"Found {len(pairs)} pairs in PD dataset")
    
    if not pairs:
        print("No pairs found!")
        return
    
    # Evaluate your inference
    your_results = evaluate_inference_method(pairs, your_localize, "Your Inference")
    
    # Evaluate Senthil's fast inference
    senthil_results = evaluate_inference_method(pairs, senthil_localize.localize, "Senthil's Fast Inference")
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    if your_results and senthil_results:
        your_success = sum(1 for r in your_results if r["success"])
        senthil_success = sum(1 for r in senthil_results if r["success"])
        total = len(your_results)
        
        your_accuracy = (your_success / total * 100)
        senthil_accuracy = (senthil_success / total * 100)
        
        your_errors = [r["error_px"] for r in your_results if "error_px" in r]
        senthil_errors = [r["error_px"] for r in senthil_results if "error_px" in r]
        
        your_times = [r["time_sec"] for r in your_results if "time_sec" in r]
        senthil_times = [r["time_sec"] for r in senthil_results if "time_sec" in r]
        
        print(f"\n{'Metric':<20} {'Your Inference':<20} {'Senthil Fast':<20} {'Winner':<10}")
        print("-" * 70)
        print(f"{'Accuracy':<20} {your_accuracy:<20.1f}% {senthil_accuracy:<20.1f}% {'YOURS' if your_accuracy > senthil_accuracy else 'TIE' if your_accuracy == senthil_accuracy else 'SENTHIL':<10}")
        
        if your_errors and senthil_errors:
            print(f"{'Mean Error':<20} {np.mean(your_errors):<20.2f}px {np.mean(senthil_errors):<20.2f}px {'YOURS' if np.mean(your_errors) < np.mean(senthil_errors) else 'TIE' if np.mean(your_errors) == np.mean(senthil_errors) else 'SENTHIL':<10}")
            print(f"{'Max Error':<20} {np.max(your_errors):<20.2f}px {np.max(senthil_errors):<20.2f}px {'YOURS' if np.max(your_errors) < np.max(senthil_errors) else 'TIE' if np.max(your_errors) == np.max(senthil_errors) else 'SENTHIL':<10}")
        
        if your_times and senthil_times:
            print(f"{'Mean Time':<20} {np.mean(your_times)*1000:<20.2f}ms {np.mean(senthil_times)*1000:<20.2f}ms {'SENTHIL' if np.mean(your_times) > np.mean(senthil_times) else 'TIE' if np.mean(your_times) == np.mean(senthil_times) else 'YOURS':<10}")

if __name__ == "__main__":
    main()