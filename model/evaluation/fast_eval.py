import os
import sys
import json
import time
import argparse
import math
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pyramid_siamese import PyramidSiameseNetwork
from inference.inference_hybrid import localize_hybrid
import cv2

def localize_ncc_only(reference_path, search_path):
    start = time.time()
    ref_img_full = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    ref_eq = cv2.equalizeHist(ref_img_100)
    search_eq = cv2.equalizeHist(search_img)
    
    ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc_result)
    
    duration = time.time() - start
    return max_loc[0] + 50, max_loc[1] + 50, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    meta_path = os.path.join(args.data_dir, "dataset_manifest.json")
    
    with open(meta_path, 'r') as f:
        manifest = json.load(f)
        samples = [r for r in manifest.get('pairs', []) if r['split'] == args.split]

    print("Loading PyTorch model (Done ONCE)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"Evaluating {len(samples)} pairs...")
    
    ncc_errors, siam_errors = [], []
    ncc_times, siam_times = [], []

    for s in samples:
        pair_id = s['pair_id']
        ref = os.path.join(args.data_dir, args.split, pair_id, "reference.png")
        search = os.path.join(args.data_dir, args.split, pair_id, "search.png")
        tx, ty = float(s['center_x']), float(s['center_y'])

        # NCC Only
        nx, ny, nt = localize_ncc_only(ref, search)
        n_err = math.sqrt((nx - tx)**2 + (ny - ty)**2)
        
        # Hybrid (NCC + Siamese)
        start_siam = time.time()
        sx, sy = localize_hybrid(model, ref, search, device, verbose=False)
        st = time.time() - start_siam
        s_err = math.sqrt((sx - tx)**2 + (sy - ty)**2)

        ncc_errors.append(n_err)
        siam_errors.append(s_err)
        ncc_times.append(nt)
        siam_times.append(st)

    ncc_acc = sum(1 for e in ncc_errors if e <= args.tolerance) / len(samples) * 100
    siam_acc = sum(1 for e in siam_errors if e <= args.tolerance) / len(samples) * 100
    
    print("\n" + "=" * 65)
    print("TRUE INFERENCE SPEED COMPARISON (No Subprocess Overhead)")
    print("=" * 65)
    print(f"Tolerance: {args.tolerance} px")
    print(f"NCC Accuracy:    {ncc_acc:.1f}% | Mean Error: {np.mean(ncc_errors):.2f}px | Avg Time: {np.mean(ncc_times)*1000:.1f}ms")
    print(f"Hybrid Accuracy: {siam_acc:.1f}% | Mean Error: {np.mean(siam_errors):.2f}px | Avg Time: {np.mean(siam_times)*1000:.1f}ms")

if __name__ == "__main__":
    main()
