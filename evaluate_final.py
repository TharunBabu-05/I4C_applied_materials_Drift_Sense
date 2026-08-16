import os
import sys
import json
import time
import math
import torch
import numpy as np

# Suppress simple warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# Import Hybrid Model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "model")))
from models.pyramid_siamese import PyramidSiameseNetwork
import importlib.util

# Import Hybrid Inference
spec_hybrid = importlib.util.spec_from_file_location("inference_hybrid", "model/inference/inference_hybrid.py")
inference_hybrid = importlib.util.module_from_spec(spec_hybrid)
spec_hybrid.loader.exec_module(inference_hybrid)
localize_hybrid = inference_hybrid.localize_hybrid

# Import Original Baseline
spec_orig = importlib.util.spec_from_file_location("inference_orig", "inference.py")
inference_orig = importlib.util.module_from_spec(spec_orig)
spec_orig.loader.exec_module(inference_orig)
localize_original = inference_orig.localize

import argparse

def evaluate_folder(data_dir, checkpoint, encoder_type, tolerance=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n--- Evaluating Model: {checkpoint} (Encoder: {encoder_type}) ---")
    print("Loading PyTorch model...")
    model = PyramidSiameseNetwork(encoder_type=encoder_type).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Collect all json ground truths recursively
    gt_files = []
    if os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            has_ref = any(f in files for f in ["reference.png", "target.png", "reference_100x.png"])
            has_search = any(f in files for f in ["search.png", "search_10x.png"])
            if has_ref and has_search:
                gt_files.append(root)
                    
    print(f"Total images found in {data_dir}: {len(gt_files)}")
    
    ncc_errors, siam_errors = [], []
    ncc_times, siam_times = [], []

    for i, pair_dir in enumerate(gt_files):
        # Dynamically find the exact filenames
        files = os.listdir(pair_dir)
        ref_name = next(f for f in files if f in ["reference.png", "target.png", "reference_100x.png"])
        search_name = next(f for f in files if f in ["search.png", "search_10x.png"])
        
        ref = os.path.join(pair_dir, ref_name)
        search = os.path.join(pair_dir, search_name)
        
        # Dynamically find GT file
        gt_name = next((f for f in files if "ground" in f.lower() or "gt" in f.lower()), None)
        if not gt_name:
            continue
        gt_path = os.path.join(pair_dir, gt_name)
            
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            if "target_x" in gt:
                gt_x, gt_y = gt["target_x"], gt["target_y"]
            elif "target" in gt and "search_center_xy" in gt["target"]:
                gt_x, gt_y = gt["target"]["search_center_xy"]
            else:
                gt_x, gt_y = float(gt.get('center_x', 500)), float(gt.get('center_y', 500))

        # 1. Evaluate Baseline NCC
        start_t = time.time()
        res_ncc = localize_original(ref, search, verbose=False)
        ncc_times.append((time.time() - start_t) * 1000)
        
        if res_ncc is not None:
            pred_x, pred_y = res_ncc
            err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            ncc_errors.append(err)
        else:
            ncc_errors.append(float('inf'))

        # 2. Evaluate Hybrid Siamese
        start_t = time.time()
        res_siam = localize_hybrid(model, ref, search, device, verbose=False)
        siam_times.append((time.time() - start_t) * 1000)
        
        if res_siam is not None:
            pred_x, pred_y = res_siam
            err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            siam_errors.append(err)
        else:
            siam_errors.append(float('inf'))

    if len(ncc_errors) == 0:
        print("No valid pairs found to evaluate.")
        return

    # Calculate metrics
    ncc_hits = sum(1 for e in ncc_errors if e <= tolerance)
    siam_hits = sum(1 for e in siam_errors if e <= tolerance)
    
    total = len(ncc_errors)
    ncc_acc = (ncc_hits / total) * 100
    siam_acc = (siam_hits / total) * 100
    
    # Exclude infinities for mean error calculation
    valid_ncc_errs = [e for e in ncc_errors if e != float('inf')]
    valid_siam_errs = [e for e in siam_errors if e != float('inf')]
    
    mean_ncc_err = sum(valid_ncc_errs) / len(valid_ncc_errs) if valid_ncc_errs else float('inf')
    mean_siam_err = sum(valid_siam_errs) / len(valid_siam_errs) if valid_siam_errs else float('inf')
    
    avg_ncc_time = sum(ncc_times) / len(ncc_times)
    avg_siam_time = sum(siam_times) / len(siam_times)

    print("\n=================================================================")
    print(f"FINAL MACHINE LEARNING METRICS (TEST SET: {total} IMAGES)")
    print("=================================================================")
    print(f"Tolerance Threshold: {tolerance} pixels\n")
    
    print("[1] INFERENCE SPEED (Average Time per Image)")
    print(f"    - Baseline NCC (inference.py) : {avg_ncc_time:.1f} ms")
    print(f"    - Hybrid Siamese Model        : {avg_siam_time:.1f} ms\n")
    
    print("[2] MEAN ERROR DISTANCE")
    print(f"    - Baseline NCC (inference.py) : {mean_ncc_err:.2f} pixels")
    print(f"    - Hybrid Siamese Model        : {mean_siam_err:.2f} pixels\n")
    
    print("[3] LOCALIZATION ACCURACY")
    print(f"    - Baseline NCC (inference.py) : {ncc_acc:.1f}%")
    print(f"    - Hybrid Siamese Model        : {siam_acc:.1f}%\n")
    
    print("[4] CONFUSION MATRIX (Localization Context)")
    print("    * True Positive (Hit) = Predicted within 5px of target")
    print("    * False Negative (Miss) = Predicted outside 5px\n")
    print(f"    Baseline NCC Matrix:\n      [ {ncc_hits:>3} Hits ]  [ {total - ncc_hits:>3} Misses ]")
    print(f"    Hybrid Siamese Matrix:\n      [ {siam_hits:>3} Hits ]  [ {total - siam_hits:>3} Misses ]")
    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="model/data/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, choices=['resnet', 'mobilenet'], default='resnet')
    args = parser.parse_args()
    
    evaluate_folder(args.data_dir, args.checkpoint, args.encoder)
