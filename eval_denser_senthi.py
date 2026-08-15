import os
import sys
import json
import time
import math
import torch
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "model")))
from models.pyramid_siamese import PyramidSiameseNetwork
import importlib.util
spec = importlib.util.spec_from_file_location("inference_hybrid", "model/inference/inference_hybrid.py")
inference_hybrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_hybrid)
localize_hybrid = inference_hybrid.localize_hybrid

import importlib.util

# Import the user's original inference.py baseline
spec_orig = importlib.util.spec_from_file_location("inference_orig", "/media/tharun/OS/Semester-7/I4C_hackathon/inference.py")
inference_orig = importlib.util.module_from_spec(spec_orig)
spec_orig.loader.exec_module(inference_orig)
localize_original = inference_orig.localize

def localize_ncc_only(reference_path, search_path):
    start = time.time()
    # Call the exact original v3.0 pipeline
    res_x, res_y = localize_original(reference_path, search_path, verbose=False)
    duration = time.time() - start
    return float(res_x), float(res_y), duration

def main():
    data_dir = "dataset_most_denser_senthi"
    checkpoint = "model/checkpoints/best_model_level1.pth"
    tolerance = 5.0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    pairs = [d for d in os.listdir(data_dir) if d.startswith("pair_")]
    pairs.sort()
    
    print(f"Evaluating {len(pairs)} pairs in {data_dir}...\n")
    print(f"{'ID':<10} | {'NCC Error':<12} | {'Siam Error':<12} | {'NCC Time':<10} | {'Siam Time':<10}")
    print("-" * 65)

    ncc_errors, siam_errors = [], []
    ncc_times, siam_times = [], []

    for pair_id in pairs:
        pair_dir = os.path.join(data_dir, pair_id)
        ref = os.path.join(pair_dir, "reference.png")
        search = os.path.join(pair_dir, "search.png")
        gt_path = os.path.join(pair_dir, "ground_truth.json")
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            tx, ty = float(gt['center_x']), float(gt['center_y'])

        # NCC Only
        nx, ny, nt = localize_ncc_only(ref, search)
        n_err = math.sqrt((nx - tx)**2 + (ny - ty)**2)
        
        # Hybrid
        start_siam = time.time()
        sx, sy = localize_hybrid(model, ref, search, device, verbose=False)
        st = time.time() - start_siam
        s_err = math.sqrt((sx - tx)**2 + (sy - ty)**2)

        ncc_errors.append(n_err)
        siam_errors.append(s_err)
        ncc_times.append(nt)
        siam_times.append(st)
        
        print(f"{pair_id:<10} | {n_err:<12.2f} | {s_err:<12.2f} | {nt*1000:<8.1f}ms | {st*1000:<8.1f}ms")

    ncc_acc = sum(1 for e in ncc_errors if e <= tolerance) / len(pairs) * 100
    siam_acc = sum(1 for e in siam_errors if e <= tolerance) / len(pairs) * 100
    
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"Tolerance: {tolerance} px")
    print(f"NCC Accuracy:    {ncc_acc:.1f}% | Mean Error: {np.mean(ncc_errors):.2f}px | Avg Time: {np.mean(ncc_times)*1000:.1f}ms")
    print(f"Hybrid Accuracy: {siam_acc:.1f}% | Mean Error: {np.mean(siam_errors):.2f}px | Avg Time: {np.mean(siam_times)*1000:.1f}ms")

if __name__ == "__main__":
    main()
