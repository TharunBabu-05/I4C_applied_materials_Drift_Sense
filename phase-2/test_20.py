import sys, cv2, numpy as np
sys.path.insert(0, 'phase-2/models')
import torch
import json, math, os, csv

data_dir = 'phase-2/phase2_generator_60/local_phase2_60gen_200_pairs'
sys.path.insert(0, 'phase-2')
from register import run_hybrid_inference, ncc_pyramid_fallback, REJECTION_THRESHOLD

chk = r'phase-2\checkpoints_phase2_tuned\best_model_phase2.pth'

n_pass, n_fail, n_absent_correct, n_absent_wrong = 0, 0, 0, 0
errors = []

for i in range(1, 21):
    pid = f'pair_{i:03d}'
    ref_p  = f'{data_dir}/{pid}/reference.png'
    srch_p = f'{data_dir}/{pid}/search.png'
    gt_p   = f'{data_dir}/{pid}/groundtruth.json'
    gt = json.load(open(gt_p))
    found_gt = gt.get('found', 1)
    
    x, y, theta, scale, found, score = run_hybrid_inference(ref_p, srch_p, chk)
    
    if found_gt == 0:
        if found == 0:
            n_absent_correct += 1
        else:
            n_absent_wrong += 1
    else:
        gx, gy = gt['target_x'], gt['target_y']
        err = math.sqrt((x-gx)**2 + (y-gy)**2)
        errors.append(err)
        ok = err <= 5
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        
        scale_gt = gt.get('scale', 10.0)
        print(f"{pid} {gt['set']} GT=({gx:.0f},{gy:.0f}) scale={scale_gt:.1f} Pred=({x:.0f},{y:.0f}) scale_pred={scale:.1f} err={err:.1f} {'OK' if ok else 'FAIL'}")

print()
print(f'Present: {n_pass} pass / {n_fail} fail = {100*n_pass/max(1,n_pass+n_fail):.0f}%')
print(f'Absent:  {n_absent_correct} correct / {n_absent_wrong} wrong')
print(f'Mean error (present): {sum(errors)/max(1,len(errors)):.1f} px')
