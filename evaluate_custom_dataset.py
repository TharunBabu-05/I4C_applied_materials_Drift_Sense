#!/usr/bin/env python3
"""
evaluate_custom_dataset.py
==========================
Generic evaluator for evaluating custom datasets (like Senthil's datasets).
Usage:
    python evaluate_custom_dataset.py <dataset_dir>
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

# ── Make sure inference.py is importable ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from inference import localize

TOLERANCE = 10   # px — pass / fail threshold

# ── Helpers ───────────────────────────────────────────────────────────────────
def euclidean(pred, gt_x, gt_y):
    return math.sqrt((pred[0] - gt_x) ** 2 + (pred[1] - gt_y) ** 2)

def save_visualization(ref_path, srch_path, pred, gt_x, gt_y,
                       error, pair_name, out_path, passed, extra_gt=None):
    ref_img  = Image.open(ref_path).convert("RGB")
    srch_img = Image.open(srch_path).convert("RGB")

    margin  = 24
    title_h = 65
    foot_h  = 30
    w = ref_img.width + srch_img.width + margin * 3
    h = max(ref_img.height, srch_img.height) + margin * 2 + title_h + foot_h

    canvas = Image.new("RGB", (w, h), (22, 24, 32))
    draw   = ImageDraw.Draw(canvas)

    ref_ox  = margin
    srch_ox = margin * 2 + ref_img.width
    img_oy  = title_h + margin

    canvas.paste(ref_img,  (ref_ox,  img_oy))
    canvas.paste(srch_img, (srch_ox, img_oy))

    stat_col = (70, 210, 110) if passed else (230, 70, 70)
    stat_txt = "PASS" if passed else "FAIL"

    draw.rectangle([(0, 0), (w, title_h - 4)], fill=(35, 38, 52))
    draw.text((margin, 7),
              f"{pair_name}  |  Custom Dataset Evaluation",
              fill=(210, 215, 235))
    draw.text((margin, 25),
              f"Error: {error:.1f} px   [{stat_txt}]   "
              f"GT=({gt_x},{gt_y})   Pred=({pred[0]},{pred[1]})",
              fill=stat_col)
    draw.text((margin, 45),
              f"Tolerance: {TOLERANCE} px   |   Downsample: 10×",
              fill=(140, 145, 165))

    draw.text((ref_ox,  img_oy - 20), "REFERENCE  (100× mag)",  fill=(150, 175, 220))
    draw.text((srch_ox, img_oy - 20), "SEARCH  (10× mag)",      fill=(150, 175, 220))

    # GT Green cross
    gx_abs = srch_ox + gt_x
    gy_abs = img_oy  + gt_y
    cs = 13
    draw.line([(gx_abs - cs, gy_abs), (gx_abs + cs, gy_abs)],
              fill=(55, 215, 75), width=2)
    draw.line([(gx_abs, gy_abs - cs), (gx_abs, gy_abs + cs)],
              fill=(55, 215, 75), width=2)

    # Predicted Red circle
    px_abs = srch_ox + pred[0]
    py_abs = img_oy  + pred[1]
    r = 11
    draw.ellipse([(px_abs - r, py_abs - r), (px_abs + r, py_abs + r)],
                 outline=(230, 60, 60), width=2)

    # Error vector
    if error > 2:
        draw.line([(gx_abs, gy_abs), (px_abs, py_abs)],
                  fill=(255, 210, 0), width=1)

    # Unique notch
    if extra_gt is not None:
        nx = srch_ox + extra_gt["x"]
        ny = img_oy  + extra_gt["y"]
        ns = 6
        draw.polygon([(nx, ny - ns), (nx + ns, ny),
                      (nx, ny + ns), (nx - ns, ny)],
                     outline=(255, 230, 0))

    # Legend
    lx = srch_ox + srch_img.width - 200
    ly = img_oy  + srch_img.height - 80
    draw.rectangle([(lx - 8, ly - 8), (lx + 194, ly + 74)], fill=(10, 10, 14))
    draw.line([(lx, ly + 8), (lx + 16, ly + 8)],   fill=(55, 215, 75), width=2)
    draw.text((lx + 22, ly + 1),  "Ground Truth center",  fill=(55, 215, 75))
    draw.ellipse([(lx + 2, ly + 24), (lx + 14, ly + 36)],
                 outline=(230, 60, 60), width=2)
    draw.text((lx + 22, ly + 24), "Predicted center",      fill=(230, 60, 60))
    draw.line([(lx, ly + 50), (lx + 16, ly + 50)], fill=(255, 210, 0), width=1)
    draw.text((lx + 22, ly + 44), "Error vector",           fill=(255, 210, 0))

    if extra_gt is not None:
        ns2 = 5
        draw.polygon([(lx + 8, ly + 60 - ns2), (lx + 8 + ns2, ly + 60),
                      (lx + 8, ly + 60 + ns2), (lx + 8 - ns2, ly + 60)],
                     outline=(255, 230, 0))
        draw.text((lx + 22, ly + 55), "Unique notch",         fill=(255, 230, 0))

    canvas.save(str(out_path))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Path to the custom dataset directory")
    args = parser.parse_args()
    
    DATASET_DIR = Path(args.dataset_dir)
    RESULTS_DIR = DATASET_DIR / "results_drift_sense"
    VIS_DIR     = RESULTS_DIR / "visualizations"
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    pair_dirs = sorted(DATASET_DIR.glob("pair_*"))
    if not pair_dirs:
        print(f"[ERROR] No pair_* directories found in {DATASET_DIR}")
        sys.exit(1)

    n_total  = len(pair_dirs)
    print("=" * 68)
    print(f"Drift-Sense  |  Evaluating: {DATASET_DIR.name}")
    print(f"  Dataset   : {DATASET_DIR.resolve()}")
    print(f"  Pairs     : {n_total}")
    print(f"  Tolerance : {TOLERANCE} px")
    print(f"  Results   : {RESULTS_DIR.resolve()}")
    print("=" * 68)

    results = []
    for pair_dir in pair_dirs:
        ref_p  = pair_dir / "reference.png"
        srch_p = pair_dir / "search.png"
        gt_p   = pair_dir / "ground_truth.json"

        if not (ref_p.exists() and srch_p.exists() and gt_p.exists()):
            continue

        with open(gt_p) as f:
            gt = json.load(f)

        gt_x = gt["center_x"]
        gt_y = gt["center_y"]
        notch = gt.get("unique_notch", None)
        pname = pair_dir.name

        print(f"\n[{pname}]  GT=({gt_x},{gt_y})", end="")
        if notch:
            print(f"  notch=({notch['x']},{notch['y']})", end="")
        print()

        t0 = time.time()
        try:
            pred = localize(str(ref_p), str(srch_p), verbose=False)
        except Exception as e:
            print(f"  [WARN] Inference error: {e}")
            pred = (500, 500)
        inf_t = time.time() - t0

        error  = euclidean(pred, gt_x, gt_y)
        passed = error <= TOLERANCE
        status = "PASS" if passed else "FAIL"

        print(f"  Pred=({pred[0]},{pred[1]})  "
              f"Error={error:.1f}px  {status}  ({inf_t:.3f}s)")

        vis_name = f"{pname}_result.png"
        save_visualization(
            ref_p, srch_p, pred, gt_x, gt_y,
            error, pname,
            VIS_DIR / vis_name,
            passed,
            extra_gt=notch,
        )
        print(f"  Visualization -> {vis_name}")

        results.append({
            "pair":     pname,
            "gt_x":     gt_x,
            "gt_y":     gt_y,
            "pred_x":   pred[0],
            "pred_y":   pred[1],
            "error_px": round(error, 2),
            "passed":   passed,
            "inf_time": round(inf_t, 3),
            "vis":      vis_name,
        })

    n_pass = sum(r["passed"] for r in results)
    acc    = n_pass / n_total * 100
    m_err  = sum(r["error_px"] for r in results) / n_total
    m_inf  = sum(r["inf_time"] for r in results) / n_total

    print("\n" + "=" * 68)
    print("EVALUATION SUMMARY")
    print("=" * 68)
    print(f"{'Pair':<12} {'GT':^12} {'Predicted':^12} {'Error':>9}  Status  Time")
    print("-" * 68)
    for r in results:
        gt_s = f"({r['gt_x']},{r['gt_y']})"
        pr_s = f"({r['pred_x']},{r['pred_y']})"
        st   = "PASS" if r["passed"] else "FAIL"
        print(f"{r['pair']:<12} {gt_s:^12} {pr_s:^12} "
              f"{r['error_px']:>8.1f}px  {st:<6}  {r['inf_time']:.3f}s")
    print("=" * 68)
    print(f"Accuracy  : {acc:.0f}%  ({n_pass}/{n_total} within {TOLERANCE}px)")
    print(f"Mean error: {m_err:.2f} px")
    print(f"Mean time : {m_inf:.3f} s/pair")
    print(f"Visuals   : {VIS_DIR.resolve()}")
    print("=" * 68)

    report = {
        "dataset":          DATASET_DIR.name,
        "inference_engine": "Drift-Sense v3.0 (Multi-Scale NCC Pyramid)",
        "accuracy_pct":     round(acc, 1),
        "mean_error_px":    round(m_err, 2),
        "mean_inference_s": round(m_inf, 3),
        "tolerance_px":     TOLERANCE,
        "n_pairs":          n_total,
        "pairs":            results,
    }
    rpt = RESULTS_DIR / "evaluation_report.json"
    with open(rpt, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report    : {rpt.resolve()}")

if __name__ == "__main__":
    main()
