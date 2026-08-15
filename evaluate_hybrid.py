#!/usr/bin/env python3
import json
import math
import sys
import time
from pathlib import Path

import torch

# Import the hybrid inference function
sys.path.insert(0, str(Path(r"c:\Semester-7\I4C_hackathon\model\inference")))
# pyrefly: ignore [missing-import]
from inference_hybrid import localize_hybrid
sys.path.insert(0, str(Path(r"c:\Semester-7\I4C_hackathon\model")))
# pyrefly: ignore [missing-import]
from models.pyramid_siamese import PyramidSiameseNetwork

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = Path(r"c:\Semester-7\I4C_hackathon\model\test_senthil")
CHECKPOINT_PATH = Path(r"c:\Semester-7\I4C_hackathon\model\checkpoints\best_model_level1.pth")
RESULTS_DIR = DATASET_DIR / "results_hybrid"
TOLERANCE   = 10   # px — pass / fail threshold

# ── Helpers ───────────────────────────────────────────────────────────────────
def euclidean(pred, gt_x, gt_y):
    return math.sqrt((pred[0] - gt_x) ** 2 + (pred[1] - gt_y) ** 2)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all pair directories
    pair_dirs = sorted(DATASET_DIR.glob("pair_*"))
    if not pair_dirs:
        print(f"[ERROR] No pair_* directories found in {DATASET_DIR}")
        sys.exit(1)

    n_total  = len(pair_dirs)
    print("=" * 68)
    print("Drift-Sense  |  Evaluating Hybrid Model on test_senthil")
    print(f"  Dataset   : {DATASET_DIR.resolve()}")
    print(f"  Pairs     : {n_total}")
    print(f"  Model     : {CHECKPOINT_PATH.resolve()}")
    print("=" * 68)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = PyramidSiameseNetwork().to(device)
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(str(CHECKPOINT_PATH), map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    results = []
    
    # Process just the first 100 for a quick evaluation, or all if preferred.
    # Let's do all 1000 since we have a timeout of ~10 mins in background tasks.
    # Actually, to make sure it finishes, we'll process 1000 but print every 50
    for i, pair_dir in enumerate(pair_dirs):
        ref_p  = pair_dir / "reference.png"
        srch_p = pair_dir / "search.png"
        gt_p   = pair_dir / "groundtruth.json"

        if not (ref_p.exists() and srch_p.exists() and gt_p.exists()):
            continue

        with open(gt_p) as f:
            gt = json.load(f)

        gt_x = gt["center_x"]
        gt_y = gt["center_y"]
        pname = pair_dir.name

        t0 = time.time()
        try:
            pred_x, pred_y = localize_hybrid(model, str(ref_p), str(srch_p), device, verbose=False)
            pred = (pred_x, pred_y)
        except Exception as e:
            print(f"  [WARN] Inference error on {pname}: {e}")
            pred = (500, 500)
        inf_t = time.time() - t0

        error  = euclidean(pred, gt_x, gt_y)
        passed = error <= TOLERANCE

        if (i+1) % 50 == 0 or i == 0:
            status = "PASS" if passed else "FAIL"
            print(f"[{i+1}/{n_total}] {pname}  GT=({gt_x},{gt_y}) Pred=({pred[0]:.1f},{pred[1]:.1f}) Err={error:.1f}px {status} ({inf_t:.3f}s)")

        results.append({
            "pair":     pname,
            "gt_x":     gt_x,
            "gt_y":     gt_y,
            "pred_x":   pred[0],
            "pred_y":   pred[1],
            "error_px": round(error, 2),
            "passed":   passed,
            "inf_time": round(inf_t, 3)
        })

    # ── Summary ────────────────────────────────────────────────────────────
    n_pass = sum(r["passed"] for r in results)
    acc    = n_pass / n_total * 100
    m_err  = sum(r["error_px"] for r in results) / n_total
    m_inf  = sum(r["inf_time"] for r in results) / n_total

    print("\n" + "=" * 68)
    print("EVALUATION SUMMARY")
    print("=" * 68)
    print(f"Accuracy  : {acc:.1f}%  ({n_pass}/{n_total} within {TOLERANCE}px)")
    print(f"Mean error: {m_err:.2f} px")
    print(f"Mean time : {m_inf:.3f} s/pair")
    print("=" * 68)

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "dataset":          "test_senthil",
        "inference_engine": "Hybrid CNN-NCC Model (best_model_level1.pth)",
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
