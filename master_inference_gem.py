#!/usr/bin/env python3
import argparse
import sys
import time
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model architecture for Hybrid Mode
try:
    from model.models.pyramid_siamese import PyramidSiameseNetwork
except ImportError:
    pass # Will be handled nicely if they don't run hybrid mode


# =============================================================================
# COMMON UTILITIES
# =============================================================================
def load_grayscale(path):
    """Load an image as grayscale uint8 array for memory efficiency."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {path}")
    return img


# =============================================================================
# BASELINE NCC (inference.py) LOGIC
# =============================================================================
def histogram_equalize(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.equalizeHist(img_uint8)

def light_denoise(image, sigma=1.0):
    ksize = int(round(sigma * 3)) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def preprocess(image, denoise_sigma=0.8):
    img = histogram_equalize(image)
    if denoise_sigma > 0:
        img = light_denoise(img, sigma=denoise_sigma)
    return img

def resize_image(image, new_size):
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode='L')
    pil = pil.resize(new_size, Image.LANCZOS)
    return np.array(pil, dtype=np.float32)

def build_pyramid_level(reference, search, ref_target_size, search_target_size=None):
    rh, rw = reference.shape
    template = resize_image(reference, (ref_target_size, ref_target_size))

    if search_target_size is not None and search_target_size != search.shape[1]:
        search_resized = resize_image(search, (search_target_size, search_target_size))
        scale_factor = search.shape[1] / search_target_size
    else:
        search_resized = search
        scale_factor = 1.0

    return template, search_resized, scale_factor

def ncc_search_baseline(search_image, template, top_k=20, min_score=-1.0):
    search_f32 = search_image.astype(np.float32)
    template_f32 = template.astype(np.float32)

    res = cv2.matchTemplate(search_f32, template_f32, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape
    sh, sw = search_image.shape

    candidates = []
    res_copy = res.copy()

    for _ in range(top_k):
        _, max_val, _, max_loc = cv2.minMaxLoc(res_copy)
        if max_val < min_score:
            break

        x, y = max_loc
        cy = y + th // 2
        cx = x + tw // 2
        candidates.append((cy, cx, float(max_val)))

        y_lo = max(0, y - th // 2)
        y_hi = min(res_copy.shape[0], y + th // 2)
        x_lo = max(0, x - tw // 2)
        x_hi = min(res_copy.shape[1], x + tw // 2)
        res_copy[y_lo:y_hi, x_lo:x_hi] = -np.inf

    return candidates

def disambiguate(candidates, image_center=(500, 500), ncc_threshold=0.05):
    if not candidates:
        return image_center
    if len(candidates) == 1:
        return (candidates[0][0], candidates[0][1])

    best_score = candidates[0][2]
    tied = [(y, x, s) for y, x, s in candidates if best_score - s < ncc_threshold]

    if len(tied) <= 1:
        return (candidates[0][0], candidates[0][1])

    cy, cx = image_center
    best = min(tied, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
    return (best[0], best[1])

def localize_baseline(reference_path, search_path, verbose=False):
    """
    Original 3-Level Pyramid NCC Baseline Inference.
    """
    start_time = time.time()

    if verbose: print("Loading images...")
    reference = load_grayscale(reference_path)
    search = load_grayscale(search_path)

    if verbose: print("Preprocessing...")
    ref_proc = preprocess(reference, denoise_sigma=0.5)
    search_proc = preprocess(search, denoise_sigma=0.8)

    # LEVEL 0
    if verbose: print("Level 0: Coarse NCC search...")
    template_l0, search_l0, scale_l0 = build_pyramid_level(ref_proc, search_proc, 50, 500)
    candidates_l0 = ncc_search_baseline(search_l0, template_l0, top_k=20, min_score=-0.5)
    candidates_l0_full = [(int(round(y * scale_l0)), int(round(x * scale_l0)), s) for y, x, s in candidates_l0]

    # LEVEL 1
    if verbose: print("Level 1: Nominal NCC refinement...")
    template_l1, search_l1, scale_l1 = build_pyramid_level(ref_proc, search_proc, 100, None)
    all_l1 = ncc_search_baseline(search_l1, template_l1, top_k=30, min_score=-0.5)

    def best_l1_near(cy, cx, radius=60):
        best = None
        best_score = -np.inf
        for y, x, s in all_l1:
            if abs(y - cy) <= radius and abs(x - cx) <= radius and s > best_score:
                best_score = s
                best = (y, x, s)
        return best

    fused_candidates = []
    for cy0, cx0, s0 in candidates_l0_full:
        c1 = best_l1_near(cy0, cx0, radius=70)
        if c1 is not None:
            y1, x1, s1 = c1
            fused_score = 0.35 * s0 + 0.65 * s1
            fused_candidates.append((y1, x1, fused_score))
        else:
            fused_candidates.append((cy0, cx0, s0 * 0.5))

    fused_candidates.sort(key=lambda c: -c[2])

    # LEVEL 2
    best_y, best_x, best_score = fused_candidates[0] if fused_candidates else (500, 500, 0)
    if verbose: print(f"Level 2: Fine refinement around ({best_x}, {best_y})...")

    win_half = 100
    wy0 = max(0, best_y - win_half)
    wy1 = min(search_proc.shape[0], best_y + win_half)
    wx0 = max(0, best_x - win_half)
    wx1 = min(search_proc.shape[1], best_x + win_half)

    search_window = search_proc[wy0:wy1, wx0:wx1]

    if search_window.shape[0] > 20 and search_window.shape[1] > 20:
        fine_scale = 2
        win_h, win_w = search_window.shape
        search_fine = resize_image(search_window, (win_w * fine_scale, win_h * fine_scale))
        template_l2 = resize_image(ref_proc, (200, 200))

        if search_fine.shape[0] >= template_l2.shape[0] and search_fine.shape[1] >= template_l2.shape[1]:
            fine_cands = ncc_search_baseline(search_fine, template_l2, top_k=5, min_score=-0.5)
            if fine_cands:
                fy, fx, fs = fine_cands[0]
                fy_full = int(round(wy0 + (fy / fine_scale)))
                fx_full = int(round(wx0 + (fx / fine_scale)))
                if fs > best_score - 0.1:
                    best_y, best_x = fy_full, fx_full

    image_center = (search_proc.shape[0] // 2, search_proc.shape[1] // 2)
    result_y, result_x = disambiguate(fused_candidates[:10], image_center=image_center, ncc_threshold=0.03)

    if abs(best_y - result_y) <= 15 and abs(best_x - result_x) <= 15:
        result_y, result_x = best_y, best_x

    result_x = max(0, min(search_proc.shape[1] - 1, int(round(result_x))))
    result_y = max(0, min(search_proc.shape[0] - 1, int(round(result_y))))

    if verbose:
        print(f"\nBaseline NCC Result: ({result_x}, {result_y})")
        print(f"Time: {(time.time() - start_time)*1000:.1f}ms")

    return result_x, result_y


# =============================================================================
# HYBRID SIAMESE LOGIC
# =============================================================================
def non_max_suppression_peaks(scores, min_distance=10, top_k=3):
    peaks = []
    score_map = scores.copy()
    for _ in range(top_k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(score_map)
        peaks.append((max_loc[0], max_loc[1], max_val))
        x, y = max_loc
        y0 = max(0, y - min_distance)
        y1 = min(score_map.shape[0], y + min_distance)
        x0 = max(0, x - min_distance)
        x1 = min(score_map.shape[1], x + min_distance)
        score_map[y0:y1, x0:x1] = -1.0
    return peaks

def localize_hybrid(reference_path, search_path, checkpoint_path, verbose=False):
    """
    Hybrid Inference using OpenCV NCC + Siamese Deep Learning Model.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose: print(f"Loading Siamese Model from {checkpoint_path}...")
    model = PyramidSiameseNetwork().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ref_img_full = load_grayscale(reference_path)
    search_img = load_grayscale(search_path)
    
    # Fast Coarse NCC Search
    ref_img_100 = cv2.resize(ref_img_full, (100, 100), interpolation=cv2.INTER_AREA)
    ref_eq = cv2.equalizeHist(ref_img_100)
    search_eq = cv2.equalizeHist(search_img)
    search_eq = cv2.GaussianBlur(search_eq, (3, 3), 1.0)
    ref_eq = cv2.GaussianBlur(ref_eq, (3, 3), 1.0)
    
    ncc_result = cv2.matchTemplate(search_eq, ref_eq, cv2.TM_CCOEFF_NORMED)
    top_peaks = non_max_suppression_peaks(ncc_result, min_distance=10, top_k=3)

    # Siamese Disambiguation
    ref_tensor = TF.to_tensor(ref_img_100).unsqueeze(0).to(device)
    candidate_patches = []
    valid_peaks = []
    
    for px, py, ncc_score in top_peaks:
        cy = py + 50
        cx = px + 50
        y0, y1 = cy - 50, cy + 50
        x0, x1 = cx - 50, cx + 50
        
        if y0 >= 0 and y1 <= search_img.shape[0] and x0 >= 0 and x1 <= search_img.shape[1]:
            crop = search_img[y0:y1, x0:x1]
            candidate_patches.append(TF.to_tensor(crop).unsqueeze(0))
            valid_peaks.append((cx, cy, ncc_score))

    if not candidate_patches:
        best_x, best_y, _ = top_peaks[0]
        return best_x + 50, best_y + 50

    batch = torch.cat(candidate_patches).to(device)
    
    with torch.no_grad():
        ref_emb = model.encoder(ref_tensor)
        batch_emb = model.encoder(batch)
        sim_scores = model.compute_similarity(ref_emb.expand(batch.size(0), -1), batch_emb).cpu().numpy()

    best_fusion_score = -1.0
    best_coord = (500, 500)
    
    for i in range(len(valid_peaks)):
        cx, cy, ncc_val = valid_peaks[i]
        siam_val = max(0.001, float(sim_scores[i]))
        ncc_val_clamped = max(0.001, float(ncc_val))
        fusion_score = 0.3 * ncc_val_clamped + 0.7 * siam_val
        
        if verbose:
            print(f"  Cand {i+1}: ({cx}, {cy}) | NCC: {ncc_val:.3f} | Siam: {siam_val:.3f} | Fused: {fusion_score:.3f}")
            
        if fusion_score > best_fusion_score:
            best_fusion_score = fusion_score
            best_coord = (int(cx), int(cy))

    if verbose:
        print(f"\nHybrid Siamese Result: {best_coord}")
        print(f"Time: {(time.time() - start_time)*1000:.1f}ms")

    return best_coord[0], best_coord[1]


# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Master Inference Script: Runs either Baseline NCC or Hybrid Siamese")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference image")
    parser.add_argument("--search", type=str, required=True, help="Path to search image")
    
    # This is the secret to differentiating them!
    parser.add_argument("--mode", type=str, choices=["baseline", "hybrid"], default="baseline",
                        help="Choose 'baseline' (Pure OpenCV) or 'hybrid' (OpenCV + Siamese AI). Default is baseline.")
    
    # Optional explicitly defined checkpoint
    parser.add_argument("--checkpoint", type=str, 
                        default="model/_resnet_final_16k_correct_Dataset_TLM/best_model_level1.pth",
                        help="Path to the trained model (only used in hybrid mode)")
    
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    
    args = parser.parse_args()

    # If the user specifically asks for hybrid, or provides a custom checkpoint, run hybrid
    if args.mode == "hybrid" or (args.checkpoint and args.checkpoint != "model/_resnet_final_16k_correct_Dataset_TLM/best_model_level1.pth" and args.mode != "baseline"):
        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Model checkpoint not found at {args.checkpoint}!")
            sys.exit(1)
            
        if args.verbose:
            print("=====================================================")
            print("RUNNING HYBRID INFERENCE (NCC + SIAMESE DISAMBIGUATION)")
            print("=====================================================")
            
        x, y = localize_hybrid(args.reference, args.search, args.checkpoint, args.verbose)
    else:
        if args.verbose:
            print("=====================================================")
            print("RUNNING BASELINE INFERENCE (PURE 3-LAYER NCC PYRAMID)")
            print("=====================================================")
            
        x, y = localize_baseline(args.reference, args.search, args.verbose)

    print(f"({x}, {y})")

if __name__ == "__main__":
    main()
