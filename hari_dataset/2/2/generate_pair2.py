import os
import sys
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ic_grid_generator import render_ic_layout, apply_ultra_heavy_sem_noise

def generate_pair2():
    pair_dir = os.path.dirname(os.path.abspath(__file__))

    tw, th = 1000, 1000

    # Ground-truth bounding box: [x, y, w, h]
    # Centered on the pair2 unique feature: the 4px jog in the vertical routing
    # channel at global coordinate (700, 300).
    # bbox [625, 225, 150, 150] → center = (700, 300) ← exactly on the jog.
    target_w, target_h = 150, 150
    gt_x, gt_y = 625, 225

    # ── Search Image ──────────────────────────────────────────────────────────
    # Render the full 1000×1000 IC layout for pair2.
    # pair_id="pair2" activates the jog in the top-right routing channel.
    search_canvas = render_ic_layout(
        canvas_size=(th, tw),
        viewport=(0.0, 1000.0, 0.0, 1000.0),
        pair_id="pair2"
    )
    search_img_uint8 = apply_ultra_heavy_sem_noise(search_canvas, noise_sigma=15.0)

    # ── Target Reference Image ────────────────────────────────────────────────
    # Render the SAME IC layout zoomed into the exact bbox region.
    # This zoomed render shows the jog feature in full detail, making the
    # target visually consistent with what appears at (gt_x, gt_y) in the search.
    target_canvas = render_ic_layout(
        canvas_size=(th, tw),
        viewport=(float(gt_x), float(gt_x + target_w),
                  float(gt_y), float(gt_y + target_h)),
        pair_id="pair2"
    )
    target_canvas_uint8 = np.clip(target_canvas, 0, 255).astype(np.uint8)
    target_blurred = cv2.GaussianBlur(target_canvas_uint8, (5, 5), 1.0)
    target_img_uint8 = apply_ultra_heavy_sem_noise(
        target_blurred.astype(np.float32), noise_sigma=4.0
    )

    # ── Save Outputs to local dir and nested pair2 subdir ────────────────────
    out_dirs = [pair_dir, os.path.join(pair_dir, "pair2")]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "search.png"), search_img_uint8)
        cv2.imwrite(os.path.join(d, "target.png"), target_img_uint8)

        gt_data = {
            "pair_id": "pair2",
            "search_image": "search.png",
            "target_image": "target.png",
            "search_size": [tw, th],
            "target_size": [tw, th],
            # [x, y, w, h] — top-left corner of the target region in the search image
            "target_bbox": [gt_x, gt_y, target_w, target_h],
            # Center of the target bbox, which coincides with the jog at (700, 300)
            "target_center": [gt_x + target_w // 2, gt_y + target_h // 2],
            "description": (
                "Design 2: Rendered IC layout with pair2 jog feature. "
                "Target bbox [625,225,150,150] is centred on the 4 px rightward jog "
                "in the vertical routing channel at global (700, 300). "
                "Search noise sigma=15, Target noise sigma=4."
            )
        }

        with open(os.path.join(d, "pair2--groundtruth.json"), 'w') as f:
            json.dump(gt_data, f, indent=4)

    print(
        f"[Pair 2] Generated search.png, target.png, and pair2--groundtruth.json\n"
        f"         GT bbox  : {gt_data['target_bbox']}\n"
        f"         GT center: {gt_data['target_center']}  <- jog at (700, 300)\n"
        f"         Output dir: {pair_dir}"
    )

if __name__ == "__main__":
    generate_pair2()
