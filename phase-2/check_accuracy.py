import json
import math

data_dir = r'phase-2\phase2_generator_60\local_phase2_60gen_200_pairs'

predictions = {
    'pair_001': (656.0, 366.0),
    'pair_002': (686.0, 352.0),
    'pair_005': (861.0, 114.0),
    'pair_010': (454.0, 550.0),
    'pair_071': (355.0, 216.0),
    'pair_181': (669.0, 377.0),
    'pair_200': (403.0, 757.0),
}

# Absent pair predictions (should be found=0)
absent_preds = {'pair_141': 0, 'pair_160': 0}

print("=" * 80)
print("Phase 2 Accuracy Check — register.py vs Ground Truth")
print("=" * 80)

errors = []
for pair_id, (px, py) in predictions.items():
    gt_path = rf'{data_dir}\{pair_id}\groundtruth.json'
    with open(gt_path) as f:
        gt = json.load(f)
    gx = gt['target_x']
    gy = gt['target_y']
    err = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
    errors.append(err)
    scale_gt = gt.get('scale', 'N/A')
    theta_gt = gt.get('theta', 'N/A')
    set_name = gt.get('set', '?')
    status = 'PASS' if err <= 5 else ('CLOSE' if err <= 10 else 'FAIL')
    print(f"[{status}] {pair_id} [{set_name}]")
    print(f"       GT=({gx:.1f},{gy:.1f})  Pred=({px:.0f},{py:.0f})  Error={err:.1f}px")
    print(f"       GT_scale={scale_gt:.2f}  GT_theta={theta_gt:.2f}")
    print()

print("-" * 40)
print("Absent pair rejection check:")
for pair_id, found_pred in absent_preds.items():
    gt_path = rf'{data_dir}\{pair_id}\groundtruth.json'
    with open(gt_path) as f:
        gt = json.load(f)
    gt_found = gt.get('found', 1)
    set_name = gt.get('set', '?')
    correct = (found_pred == gt_found)
    print(f"  [{set_name}] {pair_id}: GT_found={gt_found}  Pred_found={found_pred}  {'CORRECT' if correct else 'WRONG'}")

print()
print("=" * 80)
mean_err = sum(errors) / len(errors)
within_5 = sum(1 for e in errors if e <= 5)
within_10 = sum(1 for e in errors if e <= 10)
print(f"Mean Error    : {mean_err:.1f} px")
print(f"Within 5px    : {within_5}/{len(errors)} ({100*within_5/len(errors):.0f}%)")
print(f"Within 10px   : {within_10}/{len(errors)} ({100*within_10/len(errors):.0f}%)")
print("=" * 80)
