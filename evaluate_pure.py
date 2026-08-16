import os
import sys
import json
import time
import math
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.models.pyramid_siamese import PyramidSiameseNetwork

# Import pure siamese inference
import importlib.util
spec_pure = importlib.util.spec_from_file_location("inference_pure", "model/inference/inference_pure_siamese.py")
inference_pure = importlib.util.module_from_spec(spec_pure)
spec_pure.loader.exec_module(inference_pure)
localize_pure = inference_pure.localize_pure_siamese

def evaluate_pure(data_dir, checkpoint, encoder_type, tolerance=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Evaluating PURE Siamese Model: {checkpoint} (Encoder: {encoder_type}) ---")
    
    model = PyramidSiameseNetwork(encoder_type=encoder_type).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    gt_files = []
    if os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            if any(f in files for f in ["reference.png", "target.png"]) and any(f in files for f in ["search.png"]):
                gt_files.append(root)
                    
    print(f"Total images found in {data_dir}: {len(gt_files)}")
    
    siam_errors, siam_times = [], []

    for i, pair_dir in enumerate(gt_files[:100]):
        files = os.listdir(pair_dir)
        ref_name = next(f for f in files if f in ["reference.png", "target.png"])
        search_name = next(f for f in files if f in ["search.png"])
        ref = os.path.join(pair_dir, ref_name)
        search = os.path.join(pair_dir, search_name)
        
        gt_name = next((f for f in files if "ground" in f.lower() or "gt" in f.lower()), None)
        if not gt_name: continue
        gt_path = os.path.join(pair_dir, gt_name)
            
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            if "target_x" in gt:
                gt_x, gt_y = gt["target_x"], gt["target_y"]
            elif "target" in gt and "search_center_xy" in gt["target"]:
                gt_x, gt_y = gt["target"]["search_center_xy"]
            else:
                gt_x, gt_y = float(gt.get('center_x', 500)), float(gt.get('center_y', 500))

        start_t = time.time()
        res_siam = localize_pure(model, ref, search, device, verbose=False)
        siam_times.append((time.time() - start_t) * 1000)
        
        if res_siam is not None:
            pred_x, pred_y = res_siam
            err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            siam_errors.append(err)
        else:
            siam_errors.append(float('inf'))
            
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{len(gt_files)} images...")

    if not siam_errors:
        print("No valid pairs found.")
        return

    siam_hits = sum(1 for e in siam_errors if e <= tolerance)
    total = len(siam_errors)
    siam_acc = (siam_hits / total) * 100
    valid_siam_errs = [e for e in siam_errors if e != float('inf')]
    mean_siam_err = sum(valid_siam_errs) / len(valid_siam_errs) if valid_siam_errs else float('inf')
    avg_siam_time = sum(siam_times) / len(siam_times)

    print("\n=================================================================")
    print(f"PURE SIAMESE METRICS (TEST SET: {total} IMAGES)")
    print("=================================================================")
    print(f"Tolerance Threshold: {tolerance} pixels\n")
    print(f"[1] INFERENCE SPEED: {avg_siam_time:.1f} ms")
    print(f"[2] MEAN ERROR:      {mean_siam_err:.2f} pixels")
    print(f"[3] ACCURACY:        {siam_acc:.1f}%")
    print(f"    [ {siam_hits:>3} Hits ]  [ {total - siam_hits:>3} Misses ]")
    print("=================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="model/data_benchmark")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--encoder", type=str, choices=['resnet', 'mobilenet'], default='mobilenet')
    args = parser.parse_args()
    evaluate_pure(args.data_dir, args.checkpoint, args.encoder)
