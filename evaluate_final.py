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

def evaluate_folder(data_dir, tolerance=5.0):
    checkpoint = "model/checkpoints/best_model_level1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading PyTorch model...")
    model = PyramidSiameseNetwork().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Collect all json ground truths across train, val, test
    gt_files = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.isdir(split_dir):
            for d in os.listdir(split_dir):
                pair_dir = os.path.join(split_dir, d)
                if os.path.isdir(pair_dir):
                    gt_files.append(pair_dir)
                    
    print(f"Total images found: {len(gt_files)}")
    
    ncc_errors, siam_errors = [], []
    ncc_times, siam_times = [], []

    for i, pair_dir in enumerate(gt_files):
        ref = os.path.join(pair_dir, "reference.png")
        search = os.path.join(pair_dir, "search.png")
        gt_path = os.path.join(pair_dir, "ground_truth.json")
        
        if not os.path.exists(gt_path):
            # Sometimes it's groundtruth.json
            gt_path = os.path.join(pair_dir, "groundtruth.json")
            if not os.path.exists(gt_path):
                continue
            
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            # Standalone generator format uses target.search_center_xy
            if "target" in gt and "search_center_xy" in gt["target"]:
                tx, ty = gt["target"]["search_center_xy"]
            else:
                tx, ty = float(gt.get('center_x', 500)), float(gt.get('center_y', 500))

        # 1. Classical Baseline
        start = time.time()
        nx, ny = localize_original(ref, search, verbose=False)
        nt = time.time() - start
        
        # 2. Hybrid Model
        start = time.time()
        sx, sy = localize_hybrid(model, ref, search, device, verbose=False)
        st = time.time() - start

        ncc_errors.append(math.sqrt((nx - tx)**2 + (ny - ty)**2))
        siam_errors.append(math.sqrt((sx - tx)**2 + (sy - ty)**2))
        ncc_times.append(nt)
        siam_times.append(st)
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(gt_files)} images...")

    # Metrics Calculation
    ncc_hits = sum(1 for e in ncc_errors if e <= tolerance)
    ncc_misses = len(ncc_errors) - ncc_hits
    siam_hits = sum(1 for e in siam_errors if e <= tolerance)
    siam_misses = len(siam_errors) - siam_hits
    
    ncc_acc = (ncc_hits / len(ncc_errors)) * 100
    siam_acc = (siam_hits / len(siam_errors)) * 100
    
    print("\n=================================================================")
    print("FINAL MACHINE LEARNING METRICS (TEST SET: 500 IMAGES)")
    print("=================================================================")
    print(f"Tolerance Threshold: {tolerance} pixels\n")
    
    print("[1] INFERENCE SPEED (Average Time per Image)")
    print(f"    - Baseline NCC (inference.py) : {np.mean(ncc_times)*1000:.1f} ms")
    print(f"    - Hybrid Siamese Model        : {np.mean(siam_times)*1000:.1f} ms\n")
    
    print("[2] MEAN ERROR DISTANCE")
    print(f"    - Baseline NCC (inference.py) : {np.mean(ncc_errors):.2f} pixels")
    print(f"    - Hybrid Siamese Model        : {np.mean(siam_errors):.2f} pixels\n")
    
    print("[3] LOCALIZATION ACCURACY")
    print(f"    - Baseline NCC (inference.py) : {ncc_acc:.1f}%")
    print(f"    - Hybrid Siamese Model        : {siam_acc:.1f}%\n")
    
    print("[4] CONFUSION MATRIX (Localization Context)")
    print("    * True Positive (Hit) = Predicted within 5px of target")
    print("    * False Negative (Miss) = Predicted outside 5px\n")
    print("    Baseline NCC Matrix:")
    print(f"      [ {ncc_hits:3d} Hits ]  [ {ncc_misses:3d} Misses ]")
    print("    Hybrid Siamese Matrix:")
    print(f"      [ {siam_hits:3d} Hits ]  [ {siam_misses:3d} Misses ]")
    print("=================================================================")

if __name__ == "__main__":
    evaluate_folder("model/data_new_test")
