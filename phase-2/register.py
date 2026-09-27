#!/usr/bin/env python3
"""
register.py — Drift-Sense Phase 2 Submission Entry Point
==========================================================
THE ONLY FILE THE ORGANIZERS WILL RUN.

Usage (exact signature required by Phase 2 spec):
    python register.py --input pairs.csv --output predictions.csv

pairs.csv columns:   pair_id, reference_path, search_path
predictions.csv:     pair_id, x, y, theta, scale, found, score

Constraints (from Phase 2 addendum):
  - 4-core x86 CPU, 8 GB RAM, no GPU, no network
  - Python 3.11, weights ship inside the zip
  - Median ≤ 5 s/pair, hard timeout 20 s → score 0
  - Every pair_id must appear exactly once. Missing row = 0 pts.

Pipeline (hybrid with NCC fallback):
  1. Load images (RGB→grayscale for Set D optical pairs)
  2. NCC coarse search → Top-3 candidates
  3. Phase 2 Siamese model → similarity score, scale, theta prediction
  4. Rejection gate: if max NCC score < REJECTION_THRESHOLD → found=0
  5. Write CSV row

If the model checkpoint cannot be loaded, the script falls back
gracefully to the pure NCC pyramid pipeline (no model needed).
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
# Checkpoint path relative to THIS script's directory
_SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_PATH = _SCRIPT_DIR / "checkpoints_phase2" / "best_model_phase2.pth"
FALLBACK_CHECKPOINT = _SCRIPT_DIR.parent / "final_model" / "best_model_level1.pth"

# Rejection: if the best NCC peak score is below this, output found=0
REJECTION_THRESHOLD = 0.30

# Scale range (Phase 2 spec)
SCALE_MIN, SCALE_MAX = 8.0, 12.0
SCALE_NOMINAL = 10.0


# =============================================================================
# Image loading helpers
# =============================================================================
def load_grayscale(path: str) -> np.ndarray:
    """Loads any image (grayscale or RGB) and returns a uint8 grayscale array."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if len(img.shape) == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


# =============================================================================
# NCC pipeline (fallback / coarse stage)
# =============================================================================
def ncc_coarse_search(search_img: np.ndarray, ref_img: np.ndarray,
                      top_k: int = 2, min_distance: int = 50,
                      scale_range: tuple = (8.0, 12.0), scale_step: float = 0.5,
                      theta_range: float = 5.0, theta_step: float = 1.0):
    
    srch_eq = cv2.equalizeHist(cv2.GaussianBlur(search_img, (3, 3), 1.0))
    ref_eq  = cv2.equalizeHist(cv2.GaussianBlur(ref_img, (3, 3), 1.0))
    
    scales = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
    angles = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    
    h_ref, w_ref = ref_eq.shape
    cx_ref, cy_ref = w_ref / 2.0, h_ref / 2.0
    
    all_peaks = []
    
    for s in scales:
        factor = 10.0 / s
        interp = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        srch_s = cv2.resize(srch_eq, (0, 0), fx=factor, fy=factor, interpolation=interp)
        
        best_scale_overall = -1.0
        best_scale_result = None
        best_scale_theta = 0.0
        
        for angle in angles:
            M   = cv2.getRotationMatrix2D((cx_ref, cy_ref), float(angle), 1.0)
            rot = cv2.warpAffine(ref_eq, M, (w_ref, h_ref), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            
            result = cv2.matchTemplate(srch_s, rot, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_scale_overall:
                best_scale_overall = float(max_val)
                best_scale_result = result.copy()
                best_scale_theta = float(angle)
                
        # Extract top K peaks for this specific scale
        score_map = best_scale_result
        
        for _ in range(top_k):
            _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
            px, py = max_loc
            
            # Center in the scaled map
            cx_s, cy_s = px + w_ref // 2, py + h_ref // 2
            
            # Map back to original search image coordinates
            cx, cy = cx_s / factor, cy_s / factor
            all_peaks.append((cx, cy, float(max_val), float(s), best_scale_theta))
            
            # Mask out neighborhood in the scaled map
            min_dist_s = int(min_distance * factor)
            y0 = max(0, py - min_dist_s)
            y1 = min(score_map.shape[0], py + min_dist_s)
            x0 = max(0, px - min_dist_s)
            x1 = min(score_map.shape[1], px + min_dist_s)
            score_map[y0:y1, x0:x1] = -1.0
            
    # Sort just so the best NCC score is at index 0 for the edge fallback
    all_peaks.sort(key=lambda x: x[2], reverse=True)
    return all_peaks


def ncc_pyramid_fallback(ref_path: str, srch_path: str):
    """
    3-level NCC pyramid — activated when model fails to load.
    Returns (x, y, confidence_score).
    """
    ref_img  = load_grayscale(ref_path)
    srch_img = load_grayscale(srch_path)

    ref_eq  = cv2.equalizeHist(cv2.GaussianBlur(ref_img,  (3, 3), 0.8))
    srch_eq = cv2.equalizeHist(cv2.GaussianBlur(srch_img, (3, 3), 0.8))

    # Level 0: 50×50 template on 500×500 search
    ref_l0  = cv2.resize(ref_eq, (50, 50), interpolation=cv2.INTER_AREA)
    srch_l0 = cv2.resize(srch_eq, (500, 500), interpolation=cv2.INTER_AREA)
    res_l0  = cv2.matchTemplate(srch_l0, ref_l0, cv2.TM_CCOEFF_NORMED)
    _, mv, _, ml = cv2.minMaxLoc(res_l0)
    cx0 = int((ml[0] + 25) * 2)
    cy0 = int((ml[1] + 25) * 2)

    # Level 1: 100×100 template on full search
    ref_l1  = cv2.resize(ref_eq, (100, 100), interpolation=cv2.INTER_AREA)
    res_l1  = cv2.matchTemplate(srch_eq, ref_l1, cv2.TM_CCOEFF_NORMED)
    max_score = float(res_l1.max())
    _, mv1, _, ml1 = cv2.minMaxLoc(res_l1)
    cx1, cy1 = ml1[0] + 50, ml1[1] + 50

    # Pick closer to image center if multiple peaks are close in score
    center = (srch_img.shape[1] // 2, srch_img.shape[0] // 2)
    d0 = (cx0 - center[0])**2 + (cy0 - center[1])**2
    d1 = (cx1 - center[0])**2 + (cy1 - center[1])**2
    best_x, best_y = (cx0, cy0) if d0 < d1 else (cx1, cy1)

    return float(best_x), float(best_y), max_score


# =============================================================================
# Hybrid Siamese inference (Phase 2 model)
# =============================================================================
_model_cache = {}

def load_model(checkpoint_path: str, encoder_type: str = "resnet"):
    """Lazily load and cache the Phase 1 model."""
    global _model_cache
    if checkpoint_path in _model_cache:
        return _model_cache[checkpoint_path]

    import torch
    import sys
    from pathlib import Path
    models_path = str((Path(__file__).parent / "models").resolve())
    if models_path not in sys.path:
        sys.path.insert(0, models_path)
    from pyramid_siamese_v2 import PyramidSiameseNetworkV2

    device = torch.device("cpu")
    model = PyramidSiameseNetworkV2(embedding_dim=128, encoder_type=encoder_type).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    _model_cache[checkpoint_path] = (model, device)
    return model, device


def run_hybrid_inference(ref_path: str, srch_path: str,
                         checkpoint: str, encoder_type: str = "resnet"):
    """
    Full Phase 2 hybrid inference.
    Returns (x, y, theta, scale, found, score).
    """
    import torch
    import torchvision.transforms.functional as TF

    ref_img  = load_grayscale(ref_path)
    srch_img = load_grayscale(srch_path)

    # ── Coarse NCC Sweep ──────────────────────────────────────────────────────
    peaks = ncc_coarse_search(srch_img, ref_img, top_k=2, min_distance=50)
    
    if not peaks:
        return 0.0, 0.0, 0.0, SCALE_NOMINAL, 0, 0.0
        
    best_ncc_score = peaks[0][2]

    if best_ncc_score < REJECTION_THRESHOLD:
        return 0.0, 0.0, 0.0, SCALE_NOMINAL, 0, float(best_ncc_score)

    # ── Load model ────────────────────────────────────────────────────────────
    model, device = load_model(checkpoint, encoder_type)

    ref_100 = cv2.resize(ref_img, (100, 100), interpolation=cv2.INTER_AREA)
    ref_tensor = TF.to_tensor(ref_100).unsqueeze(0).to(device)  # [1,1,100,100]

    candidate_patches = []
    valid_peaks = []

    for cx, cy, ncc_score, ncc_scale, ncc_theta in peaks:
        patch_size = int(round(1000.0 / ncc_scale))
        half_p = patch_size // 2
        y0, y1 = int(cy - half_p), int(cy + patch_size - half_p)
        x0, x1 = int(cx - half_p), int(cx + patch_size - half_p)
        
        # Only process if completely within the image
        if 0 <= y0 and y1 <= srch_img.shape[0] and 0 <= x0 and x1 <= srch_img.shape[1]:
            crop = srch_img[y0:y1, x0:x1]
            
            # 1. Undo scale: Resize to 100x100
            interp = cv2.INTER_AREA if patch_size > 100 else cv2.INTER_CUBIC
            crop_100 = cv2.resize(crop, (100, 100), interpolation=interp)
            
            # 2. Undo rotation: Rotate by -angle
            if abs(ncc_theta) > 1e-3:
                M = cv2.getRotationMatrix2D((50, 50), -ncc_theta, 1.0)
                crop_100 = cv2.warpAffine(crop_100, M, (100, 100), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            candidate_patches.append(TF.to_tensor(crop_100).unsqueeze(0))
            valid_peaks.append((cx, cy, ncc_score, ncc_scale, ncc_theta))

    if not candidate_patches:
        # Edge fallback
        cx, cy, _, ncc_scale, ncc_theta = peaks[0]
        return float(cx), float(cy), float(ncc_theta), float(ncc_scale), 1, float(best_ncc_score)

    batch = torch.cat(candidate_patches).to(device)   # [K, 1, 100, 100]

    with torch.no_grad():
        ref_emb   = model.encoder(ref_tensor)          # [1, D]
        batch_emb = model.encoder(batch)               # [K, D]

        # Similarity scores (Phase 1 model)
        sim_scores = model.compute_similarity(
            ref_emb.expand(batch.size(0), -1), batch_emb
        ).cpu().numpy()
        
        # Phase 2 V2 model: Predict scale and theta
        if hasattr(model, "predict_pose"):
            scale_preds, theta_preds = model.predict_pose(ref_emb.expand(batch.size(0), -1), batch_emb)
            scale_preds = scale_preds.cpu().numpy()
            theta_preds = theta_preds.cpu().numpy()
        else:
            scale_preds = [None] * len(valid_peaks)
            theta_preds = [None] * len(valid_peaks)

    # ── Fusion ────────────────────────────────────────────────────────────────
    best_fusion = -1.0
    best_result = None

    for i, (cx, cy, ncc_score, ncc_scale, ncc_theta) in enumerate(valid_peaks):
        siam_val = float(sim_scores[i])
        fusion_score = 0.3 * float(ncc_score) + 0.7 * siam_val

        if fusion_score > best_fusion:
            best_fusion = fusion_score
            
            # Use model's predicted scale and theta if available, else fallback to NCC's
            final_scale = float(scale_preds[i]) if scale_preds[i] is not None else float(ncc_scale)
            final_theta = float(theta_preds[i]) if theta_preds[i] is not None else float(ncc_theta)
            
            best_result = {
                "x":     float(cx),
                "y":     float(cy),
                "theta": final_theta,
                "scale": final_scale,
                "found": 1,
                "score": float(best_fusion)
            }

    r = best_result
    return r["x"], r["y"], r["theta"], r["scale"], 1, r["score"]



# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Drift-Sense Phase 2 — Batch Inference Entry Point"
    )
    parser.add_argument("--input",    type=str, required=True,
                        help="Path to pairs.csv (pair_id, reference_path, search_path)")
    parser.add_argument("--output",   type=str, required=True,
                        help="Path to write predictions.csv")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH),
                        help="Phase 2 model checkpoint (.pth)")
    parser.add_argument("--encoder",  type=str, default="resnet",
                        choices=["resnet", "mobilenet"])
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    use_model = False
    checkpoint = args.checkpoint

    # Add this script's dir to path so models/ can be imported
    sys.path.insert(0, str(_SCRIPT_DIR))

    if os.path.exists(checkpoint):
        try:
            load_model(checkpoint, args.encoder)
            use_model = True
            print(f"[OK] Phase 2 model loaded: {checkpoint}")
        except Exception as e:
            print(f"[WARN] Failed to load Phase 2 model: {e}")
    
    if not use_model and os.path.exists(str(FALLBACK_CHECKPOINT)):
        # Try Phase 1 checkpoint (only has encoder, no pose heads)
        print(f"[INFO] Falling back to Phase 1 checkpoint (NCC pyramid pipeline)")

    if not use_model:
        print("[INFO] Using pure NCC pyramid fallback (no model)")

    # ── Read pairs.csv ────────────────────────────────────────────────────────
    pairs = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(row)

    print(f"Processing {len(pairs)} pairs...")

    # ── Predict ───────────────────────────────────────────────────────────────
    rows = []
    for i, pair in enumerate(pairs):
        pair_id   = pair["pair_id"]
        ref_path  = pair["reference_path"]
        srch_path = pair["search_path"]

        t0 = time.time()
        try:
            if use_model:
                x, y, theta, scale, found, score = run_hybrid_inference(
                    ref_path, srch_path, checkpoint, args.encoder
                )
            else:
                x, y, score = ncc_pyramid_fallback(ref_path, srch_path)
                found = 1 if score >= REJECTION_THRESHOLD else 0
                theta, scale = 0.0, SCALE_NOMINAL
                if not found:
                    x = y = 0.0

        except Exception as e:
            print(f"  [ERROR] {pair_id}: {e}")
            x, y, theta, scale, found, score = 0.0, 0.0, 0.0, 0.0, 0, 0.0

        elapsed = time.time() - t0

        rows.append({
            "pair_id": pair_id,
            "x":       round(x, 4),
            "y":       round(y, 4),
            "theta":   round(theta, 4),
            "scale":   round(scale, 4),
            "found":   int(found),
            "score":   round(score, 6),
        })

        if args.verbose or (i + 1) % 10 == 0:
            status = "PRESENT" if found else "ABSENT"
            print(f"  [{i+1}/{len(pairs)}] {pair_id}: ({x:.1f}, {y:.1f}) "
                  f"θ={theta:.2f}° s={scale:.2f} [{status}] "
                  f"conf={score:.3f}  {elapsed:.3f}s")

    # ── Write predictions.csv ─────────────────────────────────────────────────
    fieldnames = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_found  = sum(1 for r in rows if r["found"] == 1)
    n_absent = len(rows) - n_found
    print(f"\nDone. Wrote {len(rows)} rows to {args.output}")
    print(f"  Present: {n_found}  |  Absent: {n_absent}")


if __name__ == "__main__":
    main()
