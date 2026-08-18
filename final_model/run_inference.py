"""
Drift-Sense: Siamese Localization Inference
============================================
Self-contained inference script for the Siamese Localization System.
Supports both pure Siamese and hybrid (NCC + Siamese) modes.

Usage:
    # Single pair inference (pure Siamese, ResNet encoder):
    python run_inference.py --reference path/to/ref.png --search path/to/search.png \
        --checkpoint best_model_level_resnet4_final.pth --encoder resnet --verbose

    # Single pair inference (hybrid mode, MobileNet encoder):
    python run_inference.py --reference path/to/ref.png --search path/to/search.png \
        --checkpoint best_model_level_mobilenet_v3.pth --encoder mobilenet --mode hybrid --verbose

    # Evaluate all 60 pairs:
    python run_inference.py --evaluate --checkpoint best_model_level_resnet4_final.pth --encoder resnet --verbose
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

# Add this directory to path so `models` package resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pyramid_siamese import PyramidSiameseNetwork


# ── Utility Functions ─────────────────────────────────────────────────

def load_grayscale(path):
    """Load an image as grayscale using OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img


def extract_patches(image, centers, patch_size=100):
    """Extract 100x100 patches centered at the given (x, y) coordinates."""
    half = patch_size // 2
    patches = []
    valid_centers = []

    h, w = image.shape
    for cx, cy in centers:
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        if y0 >= 0 and y1 <= h and x0 >= 0 and x1 <= w:
            crop = image[y0:y1, x0:x1]
            patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_centers.append((cx, cy))

    if not patches:
        return None, []
    return torch.cat(patches), valid_centers


def non_max_suppression_peaks(scores, min_distance=10, top_k=3):
    """Find top-K peaks in a 2D score map with non-max suppression."""
    peaks = []
    score_map = scores.copy()
    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
        peaks.append((max_loc[0], max_loc[1], max_val))
        x, y = max_loc
        y0 = max(0, y - min_distance)
        y1 = min(score_map.shape[0], y + min_distance)
        x0 = max(0, x - min_distance)
        x1 = min(score_map.shape[1], x + min_distance)
        score_map[y0:y1, x0:x1] = -1.0
    return peaks


# ── Pure Siamese Localization ─────────────────────────────────────────

def localize_pure_siamese(model, reference_path, search_path, device, verbose=False):
    """Two-phase sliding-window search: coarse (stride 20) → fine (stride 1)."""
    start_time = time.time()

    ref_img_full = load_grayscale(reference_path)
    search_img = load_grayscale(search_path)

    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)

        # Phase 1: Coarse Search (stride 20)
        t_coarse = time.time()
        coarse_centers = []
        stride, half = 20, 50
        h, w = search_img.shape
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                coarse_centers.append((x, y))

        coarse_patches, valid_coarse = extract_patches(search_img, coarse_centers, 100)
        best_cx, best_cy, best_sim = 500, 500, -float('inf')

        if coarse_patches is not None:
            coarse_patches = coarse_patches.to(device)
            for i in range(0, coarse_patches.size(0), 512):
                batch = coarse_patches[i:i+512]
                batch_emb = model.encoder(batch)
                sims = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
                max_idx = np.argmax(sims)
                if sims[max_idx] > best_sim:
                    best_sim = sims[max_idx]
                    best_cx, best_cy = valid_coarse[i + max_idx]

        if verbose:
            print(f"  Coarse: ({best_cx}, {best_cy}) in {(time.time()-t_coarse)*1000:.1f}ms")

        # Phase 2: Fine Search (stride 1, radius 20 around best coarse)
        t_fine = time.time()
        fine_centers = []
        radius = 20
        for y in range(best_cy - radius, best_cy + radius + 1):
            for x in range(best_cx - radius, best_cx + radius + 1):
                fine_centers.append((x, y))

        fine_patches, valid_fine = extract_patches(search_img, fine_centers, 100)
        if fine_patches is not None:
            fine_patches = fine_patches.to(device)
            for i in range(0, fine_patches.size(0), 512):
                batch = fine_patches[i:i+512]
                batch_emb = model.encoder(batch)
                sims = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
                max_idx = np.argmax(sims)
                if sims[max_idx] > best_sim:
                    best_sim = sims[max_idx]
                    best_cx, best_cy = valid_fine[i + max_idx]

        if verbose:
            print(f"  Fine:   ({best_cx}, {best_cy}) in {(time.time()-t_fine)*1000:.1f}ms")

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Result: ({best_cx}, {best_cy})  [{elapsed*1000:.1f}ms total]")
    return float(best_cx), float(best_cy)


# ── Hybrid (NCC + Siamese) Localization ───────────────────────────────

def localize_hybrid(model, reference_path, search_path, device, verbose=False):
    """Fast NCC shortlist → Siamese re-ranking with fusion scoring."""
    start_time = time.time()

    ref_img_full = load_grayscale(reference_path)
    search_img = load_grayscale(search_path)

    # NCC coarse search
    t_ncc = time.time()
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    ref_eq = cv2.GaussianBlur(cv2.equalizeHist(ref_img_100), (3, 3), 1.0)
    search_eq = cv2.GaussianBlur(cv2.equalizeHist(search_img), (3, 3), 1.0)
    ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)
    top_peaks = non_max_suppression_peaks(ncc_result, min_distance=10, top_k=3)
    if verbose:
        print(f"  NCC:    {(time.time()-t_ncc)*1000:.1f}ms")

    # Siamese re-ranking
    t_cnn = time.time()
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)
    candidate_patches, valid_peaks = [], []

    for px, py, ncc_score in top_peaks:
        cy, cx = py + 50, px + 50
        y0, y1, x0, x1 = cy - 50, cy + 50, cx - 50, cx + 50
        if y0 >= 0 and y1 <= search_img.shape[0] and x0 >= 0 and x1 <= search_img.shape[1]:
            crop = search_img[y0:y1, x0:x1]
            candidate_patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_peaks.append((cx, cy, ncc_score))

    if not candidate_patches:
        bx, by, _ = top_peaks[0]
        return float(bx + 50), float(by + 50)

    batch = torch.cat(candidate_patches).to(device)
    model.eval()
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)
        batch_emb = model.encoder(batch)
        sim_scores = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()
    if verbose:
        print(f"  CNN:    {(time.time()-t_cnn)*1000:.1f}ms")

    # Fusion: 0.3*NCC + 0.7*Siamese
    best_fusion, best_coord = -1.0, (500, 500)
    for i, (cx, cy, ncc_val) in enumerate(valid_peaks):
        siam_val = max(0.001, float(sim_scores[i]))
        ncc_clamped = max(0.001, float(ncc_val))
        fusion = 0.3 * ncc_clamped + 0.7 * siam_val
        if verbose:
            print(f"    Cand {i+1}: ({cx},{cy}) NCC={ncc_val:.3f} Siam={siam_val:.3f} Fused={fusion:.3f}")
        if fusion > best_fusion:
            best_fusion = fusion
            best_coord = (float(cx), float(cy))

    elapsed = time.time() - start_time
    if verbose:
        print(f"  Result: {best_coord}  [{elapsed*1000:.1f}ms total]")
    return best_coord


# ── Batch Evaluation on all_60_pairs ──────────────────────────────────

def evaluate_all_60(model, device, mode, verbose):
    """Run inference on all 60 pairs and compute localization error."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "all_60_pairs")

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset not found at {dataset_dir}")
        sys.exit(1)

    pair_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("pair_")
    ])

    print(f"\n{'='*70}")
    print(f"  Drift-Sense Evaluation — {len(pair_dirs)} pairs | Mode: {mode}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")

    errors = []
    localize_fn = localize_pure_siamese if mode == "pure" else localize_hybrid

    for pair_name in pair_dirs:
        pair_path = os.path.join(dataset_dir, pair_name)
        ref_path = os.path.join(pair_path, "reference.png")
        search_path = os.path.join(pair_path, "search.png")
        gt_path = os.path.join(pair_path, "groundtruth.json")

        if not all(os.path.exists(p) for p in [ref_path, search_path, gt_path]):
            print(f"  [SKIP] {pair_name}: missing files")
            continue

        with open(gt_path, "r") as f:
            gt = json.load(f)
        gt_x, gt_y = gt["center_x"], gt["center_y"]

        if verbose:
            print(f"  [{pair_name}]")

        if mode == "pure":
            pred_x, pred_y = localize_fn(model, ref_path, search_path, device, verbose)
        else:
            pred_x, pred_y = localize_fn(model, ref_path, search_path, device, verbose)

        error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        errors.append(error)

        status = "✓" if error < 50 else "✗"
        print(f"  {status} {pair_name}: pred=({pred_x:.0f},{pred_y:.0f}) gt=({gt_x},{gt_y}) err={error:.1f}px")

    # Summary
    errors = np.array(errors)
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Pairs evaluated : {len(errors)}")
    print(f"  Mean error      : {errors.mean():.2f} px")
    print(f"  Median error    : {np.median(errors):.2f} px")
    print(f"  Std error       : {errors.std():.2f} px")
    print(f"  Max error       : {errors.max():.2f} px")
    print(f"  < 25px accuracy : {(errors < 25).sum()}/{len(errors)} ({(errors < 25).mean()*100:.1f}%)")
    print(f"  < 50px accuracy : {(errors < 50).sum()}/{len(errors)} ({(errors < 50).mean()*100:.1f}%)")
    print(f"{'='*70}\n")


# ── Main Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense Siamese Localization — Inference & Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode
    parser.add_argument("--mode", type=str, choices=["pure", "hybrid"], default="pure",
                        help="Inference mode: 'pure' (sliding window) or 'hybrid' (NCC+Siamese). Default: pure")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate all 60 pairs from all_60_pairs/ directory")

    # Single-pair inference
    parser.add_argument("--reference", type=str, help="Path to reference image (100x magnification)")
    parser.add_argument("--search", type=str, help="Path to search image (10x magnification)")

    # Model
    parser.add_argument("--checkpoint", type=str, default="best_model_level1.pth",
                        help="Path to .pth checkpoint file. Default: best_model_level1.pth")
    parser.add_argument("--encoder", type=str, choices=["resnet", "mobilenet"], default="resnet",
                        help="Encoder backbone: 'resnet' or 'mobilenet'. Default: resnet")

    # Output
    parser.add_argument("--verbose", action="store_true", help="Print detailed timing and candidate info")

    args = parser.parse_args()

    # Validation
    if not args.evaluate and (not args.reference or not args.search):
        parser.error("Either --evaluate or both --reference and --search are required.")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork(encoder_type=args.encoder).to(device)

    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded checkpoint: {ckpt_path}")
        else:
            print(f"WARNING: Checkpoint not found at {ckpt_path}, using random weights!")

    if args.evaluate:
        evaluate_all_60(model, device, args.mode, args.verbose)
    else:
        localize_fn = localize_pure_siamese if args.mode == "pure" else localize_hybrid
        if args.mode == "pure":
            x, y = localize_fn(model, args.reference, args.search, device, args.verbose)
        else:
            result = localize_fn(model, args.reference, args.search, device, args.verbose)
            x, y = result[0], result[1]
        print(f"\nPredicted location: ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
