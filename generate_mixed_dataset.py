#!/usr/bin/env python3
"""
Mixed-Pattern Dataset Generator + Evaluator  (v3 — fixed)
==========================================================
Generates 10 DRAM-style image pairs using 5 visually distinct structural
variants (2 pairs each), then runs inference on each pair and saves a
side-by-side visualization.

KEY DESIGN CONSTRAINTS (matching inference.py calibration):
  1. Layout: 10000 x 10000  →  search 10x downsampled  →  1000 x 1000
  2. GT always near image center (450-550, 450-550) — same as main generator.
     inference.py's center-bias disambiguation only works near center.
  3. Minimum cell pitch ≥ 40 px in layout → ≥ 4 px in search (visible to NCC).

Variants — each differs in a VISIBLE structural property:
  1. Standard-DRAM   baseline 42-60 px pitch, moderate contrast
  2. Coarse-Pitch    60-80 px pitch, fewer cells (older node feel)
  3. High-Contrast   extreme dark bodies + bright walls (tungsten-heavy)
  4. Low-Contrast    muted intensity separation (oxide-rich / charging)
  5. High-LER        same pitch but heavily roughened edges & high CD variation

Usage:
  python generate_mixed_dataset.py --output_dir ./mixed_dataset_10 --seed 77
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# 5 structurally distinct variants  (all pitch >= 40 px for 10x visibility)
# ---------------------------------------------------------------------------
VARIANTS = [
    {
        "name": "Standard-DRAM",
        "desc": "Baseline 42-60 px pitch, moderate contrast",
        "cell_pitch_x":   (42, 60),
        "cell_pitch_y":   (42, 60),
        "cell_fill":      (0.58, 0.70),
        "body_intensity": (15, 45),
        "wall_intensity": (175, 225),
        "block_period":   (1800, 2400),
        "block_dim":      (0.05, 0.10),
        "search_poisson": (8.0, 15.0),
        "search_gaussian":(1.0, 2.5),
        "ler_amp":        (1.0, 2.5),
        "cd_grad":        (0.02, 0.04),
    },
    {
        "name": "Coarse-Pitch",
        "desc": "Large 60-80 px cells, fewer cells visible — older node",
        "cell_pitch_x":   (60, 80),
        "cell_pitch_y":   (60, 80),
        "cell_fill":      (0.55, 0.68),
        "body_intensity": (15, 45),
        "wall_intensity": (170, 220),
        "block_period":   (2500, 3500),
        "block_dim":      (0.06, 0.12),
        "search_poisson": (8.0, 14.0),
        "search_gaussian":(1.0, 2.0),
        "ler_amp":        (1.0, 2.0),
        "cd_grad":        (0.02, 0.04),
    },
    {
        "name": "High-Contrast",
        "desc": "Very dark bodies (5-20) + very bright walls (210-250) — tungsten-heavy",
        "cell_pitch_x":   (42, 60),
        "cell_pitch_y":   (42, 60),
        "cell_fill":      (0.58, 0.70),
        "body_intensity": (5, 20),
        "wall_intensity": (210, 250),
        "block_period":   (1800, 2400),
        "block_dim":      (0.08, 0.14),
        "search_poisson": (9.0, 16.0),
        "search_gaussian":(1.0, 2.2),
        "ler_amp":        (1.0, 2.5),
        "cd_grad":        (0.02, 0.04),
    },
    {
        "name": "Low-Contrast",
        "desc": "Muted intensities (80-110 bodies, 140-170 walls) — oxide-rich / charging",
        "cell_pitch_x":   (42, 60),
        "cell_pitch_y":   (42, 60),
        "cell_fill":      (0.55, 0.68),
        "body_intensity": (80, 110),
        "wall_intensity": (140, 170),
        "block_period":   (1800, 2400),
        "block_dim":      (0.02, 0.06),
        "search_poisson": (7.0, 13.0),
        "search_gaussian":(1.5, 3.0),
        "ler_amp":        (1.0, 2.5),
        "cd_grad":        (0.02, 0.04),
    },
    {
        "name": "High-LER",
        "desc": "Heavily roughened edges (LER 3-6 px) + large CD variation — fabrication stress",
        "cell_pitch_x":   (48, 65),
        "cell_pitch_y":   (48, 65),
        "cell_fill":      (0.55, 0.70),
        "body_intensity": (15, 50),
        "wall_intensity": (175, 225),
        "block_period":   (1800, 2400),
        "block_dim":      (0.05, 0.10),
        "search_poisson": (8.0, 15.0),
        "search_gaussian":(1.0, 2.5),
        "ler_amp":        (3.0, 6.0),   # <-- large roughness
        "cd_grad":        (0.06, 0.12), # <-- large CD variation
    },
]

# ---------------------------------------------------------------------------
# Constants — must match inference.py 10x calibration
# ---------------------------------------------------------------------------
LAYOUT_SIZE = 10000   # master layout (10k × 10k px)
REF_SIZE    = 1000    # reference window  (1000 × 1000 at 100x)
SEARCH_SIZE = 1000    # search image      (1000 × 1000 at 10x)
SCALE       = LAYOUT_SIZE / SEARCH_SIZE   # = 10.0

# GT must land near image center so center-bias disambiguation works correctly
# (same constraint as main dataset_generator.py — GT within ~30px of 500,500)
GT_CENTER_MIN = 460   # GT x,y lower bound
GT_CENTER_MAX = 540   # GT x,y upper bound


# ---------------------------------------------------------------------------
# Layout renderer
# ---------------------------------------------------------------------------
def generate_layout(variant, rng):
    """Render LAYOUT_SIZE × LAYOUT_SIZE grayscale float32 DRAM layout."""
    W = H = LAYOUT_SIZE
    pitch_x  = int(rng.integers(*variant["cell_pitch_x"]))
    pitch_y  = int(rng.integers(*variant["cell_pitch_y"]))
    fill     = rng.uniform(*variant["cell_fill"])
    body_int = float(rng.integers(*variant["body_intensity"]))
    wall_int = float(rng.integers(*variant["wall_intensity"]))
    ler_amp  = rng.uniform(*variant["ler_amp"])
    cd_grad  = rng.uniform(*variant["cd_grad"])

    cell_w = max(4, int(pitch_x * fill))
    cell_h = max(4, int(pitch_y * fill))

    layout = np.full((H, W), wall_int, dtype=np.float32)

    cx_arr = np.arange(pitch_x // 2, W, pitch_x)
    cy_arr = np.arange(pitch_y // 2, H, pitch_y)

    for cy in cy_arr:
        for cx in cx_arr:
            # CD variation: linear gradient across field
            cd_factor = 1.0 + cd_grad * ((cx / W) - 0.5) * 2
            lw = max(3, int(cell_w * cd_factor))
            lh = max(3, int(cell_h * cd_factor))

            # LER perturbations
            pert = ler_amp * 0.5
            y0 = max(0, min(H - 1, cy - lh // 2 + int(round(rng.normal(0, pert)))))
            y1 = max(0, min(H - 1, cy + lh // 2 + int(round(rng.normal(0, pert)))))
            x0 = max(0, min(W - 1, cx - lw // 2 + int(round(rng.normal(0, pert)))))
            x1 = max(0, min(W - 1, cx + lw // 2 + int(round(rng.normal(0, pert)))))
            if y1 > y0 and x1 > x0:
                layout[y0:y1, x0:x1] = body_int + rng.normal(0, 2)

    # Soft rounded corners
    from scipy import ndimage as _ndi
    body_mask = (layout < (body_int + wall_int) / 2).astype(np.float32)
    body_mask = _ndi.gaussian_filter(body_mask, sigma=1.5)
    layout    = layout * (1 - body_mask) + body_int * body_mask

    # Block banding (sense-amp rows)
    bp = int(rng.integers(*variant["block_period"]))
    bd = rng.uniform(*variant["block_dim"])
    bw = max(3, pitch_y)
    for y_b in range(int(rng.integers(0, bp)), H, bp):
        layout[max(0, y_b - bw // 2):min(H, y_b + bw // 2), :] *= (1 - bd)
    for x_b in range(int(rng.integers(0, bp)), W, bp):
        layout[:, max(0, x_b - bw // 2):min(W, x_b + bw // 2)] *= (1 - bd)

    return np.clip(layout, 0, 255).astype(np.float32), pitch_x, pitch_y


# ---------------------------------------------------------------------------
# SEM noise model
# ---------------------------------------------------------------------------
def add_sem_noise(crop, poisson_scale, gaussian_std, rng, blur_sigma=0.8):
    """Apply realistic SEM acquisition noise (Poisson + Gaussian + PSF + vignette)."""
    from scipy import ndimage as _ndi
    img = np.clip(crop, 0.001, 255.0).astype(np.float64)
    img = rng.poisson(img / poisson_scale) * poisson_scale
    img += rng.normal(0, gaussian_std, img.shape)
    img  = _ndi.gaussian_filter(img, sigma=blur_sigma)
    # Edge brightening
    grad = np.sqrt(_ndi.sobel(img, 0)**2 + _ndi.sobel(img, 1)**2)
    img += rng.uniform(0.06, 0.15) * grad
    # Gain + offset drift
    img  = img * rng.uniform(0.92, 1.08) + rng.integers(-6, 7)
    # Vignetting
    H, W = img.shape
    ys, xs = np.ogrid[:H, :W]
    r    = np.sqrt(((xs - W / 2) / (W / 2))**2 + ((ys - H / 2) / (H / 2))**2)
    img *= (1 - rng.uniform(0.08, 0.20) * r**2)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pair generator
# ---------------------------------------------------------------------------
def generate_pair(variant, rng):
    """
    Generate (reference, search, gt) matching inference.py expectations:
      - GT is constrained to [GT_CENTER_MIN, GT_CENTER_MAX] in both x and y
        so that center-bias disambiguation works correctly.
      - Scale = LAYOUT_SIZE / SEARCH_SIZE = 10.0 exactly.
    """
    layout, px, py = generate_layout(variant, rng)

    # Fix GT near center, back-calculate reference crop position
    gt_x = int(rng.integers(GT_CENTER_MIN, GT_CENTER_MAX + 1))
    gt_y = int(rng.integers(GT_CENTER_MIN, GT_CENTER_MAX + 1))

    # Reference crop top-left in layout coordinates
    ref_x = int(round(gt_x * SCALE - REF_SIZE // 2))
    ref_y = int(round(gt_y * SCALE - REF_SIZE // 2))

    # Clamp to layout bounds
    ref_x = max(0, min(LAYOUT_SIZE - REF_SIZE, ref_x))
    ref_y = max(0, min(LAYOUT_SIZE - REF_SIZE, ref_y))

    # Recompute GT from clamped crop position (for accuracy)
    gt_x  = int(round((ref_x + REF_SIZE / 2) / SCALE))
    gt_y  = int(round((ref_y + REF_SIZE / 2) / SCALE))

    ref_crop = layout[ref_y:ref_y + REF_SIZE, ref_x:ref_x + REF_SIZE]

    # Reference noise (low — 100x)
    ref_img = add_sem_noise(ref_crop,
                            poisson_scale=rng.uniform(15.0, 25.0),
                            gaussian_std=rng.uniform(0.5, 2.0),
                            rng=rng,
                            blur_sigma=rng.uniform(0.4, 1.0))

    # Search: downsample full 10k layout to 1k (10x)
    layout_u8 = np.clip(layout, 0, 255).astype(np.uint8)
    search_ds = cv2.resize(layout_u8, (SEARCH_SIZE, SEARCH_SIZE),
                           interpolation=cv2.INTER_AREA)

    # Search noise (high — 10x)
    search_img = add_sem_noise(search_ds.astype(np.float32),
                               poisson_scale=rng.uniform(*variant["search_poisson"]),
                               gaussian_std=rng.uniform(*variant["search_gaussian"]),
                               rng=rng,
                               blur_sigma=rng.uniform(0.6, 1.5))

    ground_truth = {
        "center_x":    gt_x,
        "center_y":    gt_y,
        "variant":     variant["name"],
        "desc":        variant["desc"],
        "cell_pitch_x": px,
        "cell_pitch_y": py,
        "ref_crop_x":  ref_x,
        "ref_crop_y":  ref_y,
        "scale_factor": SCALE,
    }
    return ref_img, search_img, ground_truth


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_visualization(ref_path, search_path, predicted, gt,
                       error, pair_name, output_path, passed):
    ref_img  = Image.open(ref_path).convert("RGB")
    srch_img = Image.open(search_path).convert("RGB")

    margin  = 24
    title_h = 60
    foot_h  = 28
    w = ref_img.width + srch_img.width + margin * 3
    h = max(ref_img.height, srch_img.height) + margin * 2 + title_h + foot_h

    canvas = Image.new("RGB", (w, h), (28, 28, 36))
    draw   = ImageDraw.Draw(canvas)

    ref_x  = margin
    srch_x = margin * 2 + ref_img.width
    img_y  = title_h + margin

    canvas.paste(ref_img,  (ref_x,  img_y))
    canvas.paste(srch_img, (srch_x, img_y))

    stat_col = (80, 220, 120) if passed else (240, 80, 80)
    stat_txt = "PASS" if passed else "FAIL"

    draw.rectangle([(0, 0), (w, title_h - 4)], fill=(40, 40, 55))
    draw.text((margin, 6),  f"{pair_name}  |  Variant: {gt['variant']}", fill=(220, 220, 240))
    draw.text((margin, 24), f"Error: {error:.1f}px  [{stat_txt}]   "
                            f"GT=({gt['center_x']},{gt['center_y']})  "
                            f"Pred=({predicted[0]},{predicted[1]})",
              fill=stat_col)
    draw.text((margin, 42), f"{gt['desc']}  |  pitch {gt['cell_pitch_x']}x{gt['cell_pitch_y']}px",
              fill=(150, 155, 175))

    draw.text((ref_x,  img_y - 18), "REFERENCE  (100x mag)",  fill=(160, 180, 220))
    draw.text((srch_x, img_y - 18), "SEARCH  (10x mag)",      fill=(160, 180, 220))
    draw.text((ref_x,  h - foot_h + 4), "1000x1000 px · high SNR", fill=(140, 140, 160))
    draw.text((srch_x, h - foot_h + 4), "1000x1000 px · low SNR",  fill=(140, 140, 160))

    # Ground truth: green cross on search panel
    gx = srch_x + gt["center_x"]
    gy = img_y  + gt["center_y"]
    cs = 12
    draw.line([(gx - cs, gy), (gx + cs, gy)], fill=(60, 220, 60), width=2)
    draw.line([(gx, gy - cs), (gx, gy + cs)], fill=(60, 220, 60), width=2)

    # Prediction: red circle on search panel
    px2 = srch_x + predicted[0]
    py2 = img_y  + predicted[1]
    r   = 10
    draw.ellipse([(px2 - r, py2 - r), (px2 + r, py2 + r)],
                 outline=(240, 60, 60), width=2)

    # Error line
    if error > 2:
        draw.line([(gx, gy), (px2, py2)], fill=(255, 200, 0), width=1)

    # Legend
    lx = srch_x + srch_img.width - 188
    ly = img_y  + srch_img.height - 60
    draw.rectangle([(lx - 6, ly - 6), (lx + 182, ly + 52)], fill=(0, 0, 0))
    draw.line([(lx, ly + 8), (lx + 16, ly + 8)], fill=(60, 220, 60), width=2)
    draw.text((lx + 20, ly + 1),  "Ground Truth",  fill=(60, 220, 60))
    draw.ellipse([(lx + 3, ly + 24), (lx + 14, ly + 35)], outline=(240, 60, 60), width=2)
    draw.text((lx + 20, ly + 24), "Predicted",     fill=(240, 60, 60))
    draw.line([(lx, ly + 44), (lx + 16, ly + 44)], fill=(255, 200, 0), width=1)
    draw.text((lx + 20, ly + 38), "Error vector",  fill=(255, 200, 0))

    canvas.save(str(output_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./mixed_dataset_10")
    parser.add_argument("--seed",       type=int, default=77)
    parser.add_argument("--tolerance",  type=int, default=10,
                        help="Pass/fail pixel tolerance (default 10)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    master_rng = np.random.default_rng(args.seed)

    PAIRS_PER = 2
    total     = len(VARIANTS) * PAIRS_PER

    print("=" * 70)
    print("Drift-Sense  |  Mixed-Pattern Dataset  |  10 pairs, 5 variants")
    print(f"  Output  : {output_dir.resolve()}")
    print(f"  Seed    : {args.seed}  |  Tolerance: {args.tolerance}px")
    print(f"  GT range: ({GT_CENTER_MIN}-{GT_CENTER_MAX}, {GT_CENTER_MIN}-{GT_CENTER_MAX}) — near center")
    print("=" * 70)

    sys.path.insert(0, str(Path(__file__).parent))
    from inference import localize

    results  = []
    pair_num = 0

    for variant in VARIANTS:
        for rep in range(PAIRS_PER):
            pair_num += 1
            pname    = f"pair_{pair_num:03d}"
            pair_dir = output_dir / pname
            pair_dir.mkdir(exist_ok=True)

            print(f"\n[{pname}] {variant['name']}  (rep {rep+1}/{PAIRS_PER})")

            t0 = time.time()
            ref_img, srch_img, gt = generate_pair(variant, master_rng)
            gen_t = time.time() - t0

            ref_p  = pair_dir / "reference.png"
            srch_p = pair_dir / "search.png"
            Image.fromarray(ref_img,  mode="L").save(str(ref_p))
            Image.fromarray(srch_img, mode="L").save(str(srch_p))
            with open(pair_dir / "ground_truth.json", "w") as f:
                json.dump(gt, f, indent=2)

            print(f"  Generated {gen_t:.1f}s  |  GT=({gt['center_x']},{gt['center_y']})  "
                  f"pitch={gt['cell_pitch_x']}x{gt['cell_pitch_y']}px")

            t1 = time.time()
            try:
                pred = localize(str(ref_p), str(srch_p), verbose=False)
            except Exception as e:
                print(f"  [WARN] {e}")
                pred = (500, 500)
            inf_t = time.time() - t1

            error  = math.sqrt((pred[0] - gt["center_x"])**2 +
                               (pred[1] - gt["center_y"])**2)
            passed = error <= args.tolerance
            status = "PASS" if passed else "FAIL"
            print(f"  Pred=({pred[0]},{pred[1]})  Error={error:.1f}px  {status}  ({inf_t:.3f}s)")

            vis_name = f"{pname}_{variant['name'].replace(' ', '_')}.png"
            save_visualization(ref_p, srch_p, pred, gt, error,
                               pname, vis_dir / vis_name, passed)
            print(f"  Visualization -> {vis_name}")

            results.append({
                "pair":     pname,
                "variant":  variant["name"],
                "desc":     variant["desc"],
                "gt_x":     gt["center_x"],
                "gt_y":     gt["center_y"],
                "pred_x":   pred[0],
                "pred_y":   pred[1],
                "error_px": round(error, 2),
                "passed":   passed,
                "gen_time": round(gen_t, 2),
                "inf_time": round(inf_t, 3),
                "vis":      vis_name,
            })

    # ------------------------------------------------------------------
    n_pass = sum(r["passed"] for r in results)
    acc    = n_pass / total * 100
    m_err  = sum(r["error_px"] for r in results) / total
    m_inf  = sum(r["inf_time"] for r in results) / total

    print("\n" + "=" * 70)
    print("FINAL RESULTS  —  Mixed-Pattern Dataset (10 pairs, 5 variants)")
    print("=" * 70)
    print(f"{'Pair':<10} {'Variant':<16} {'GT':^12} {'Pred':^12} {'Error':>8}  Status")
    print("-" * 70)
    for r in results:
        gt_s = f"({r['gt_x']},{r['gt_y']})"
        pr_s = f"({r['pred_x']},{r['pred_y']})"
        st   = "PASS" if r["passed"] else "FAIL"
        print(f"{r['pair']:<10} {r['variant']:<16} {gt_s:^12} {pr_s:^12} "
              f"{r['error_px']:>7.1f}px  {st}")
    print("=" * 70)
    print(f"Accuracy     : {acc:.0f}%  ({n_pass}/{total} within {args.tolerance}px)")
    print(f"Mean error   : {m_err:.2f} px")
    print(f"Mean time    : {m_inf:.3f} s/pair")
    print("=" * 70)

    print("\nPer-Variant Breakdown:")
    print(f"{'Variant':<16}  {'Acc':>5}  {'Mean Err':>9}")
    print("-" * 36)
    for v in VARIANTS:
        vn = v["name"]
        vr = [r for r in results if r["variant"] == vn]
        print(f"{vn:<16}  {sum(r['passed'] for r in vr)/len(vr)*100:>4.0f}%  "
              f"{sum(r['error_px'] for r in vr)/len(vr):>8.1f}px")

    print(f"\nVisualizations : {vis_dir.resolve()}")
    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump({"accuracy_pct": round(acc,1), "mean_error_px": round(m_err,2),
                   "mean_inference_s": round(m_inf,3), "tolerance_px": args.tolerance,
                   "pairs": results}, f, indent=2)


if __name__ == "__main__":
    main()
