import os
import sys
import json
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ic_grid_generator import render_ic_layout, apply_ultra_heavy_sem_noise

def generate_pair3():
    pair_dir = os.path.dirname(os.path.abspath(__file__))

    tw, th = 1000, 1000
    target_w, target_h = 150, 150
    gt_x, gt_y = 225, 625

    # 1. Render Full 1000x1000 Search Image (Clean Canvas)
    search_canvas = render_ic_layout(canvas_size=(th, tw), viewport=(0.0, 1000.0, 0.0, 1000.0), pair_id="pair3")
    
    # Apply ultra-heavy SEM noise to search image (sigma = 95.0)
    search_img_uint8 = apply_ultra_heavy_sem_noise(search_canvas, noise_sigma=95.0)

    # 2. Render 1000x1000 Target Reference Image (Zoomed In on [225, 625, 150, 150])
    target_canvas = render_ic_layout(canvas_size=(th, tw), viewport=(float(gt_x), float(gt_x + target_w), float(gt_y), float(gt_y + target_h)), pair_id="pair3")
    
    # Add minor SEM effects and Gaussian blur for reference template
    target_canvas_uint8 = np.clip(target_canvas, 0, 255).astype(np.uint8)
    target_blurred = cv2.GaussianBlur(target_canvas_uint8, (5, 5), 1.0)
    target_img_uint8 = apply_ultra_heavy_sem_noise(target_blurred.astype(np.float32), noise_sigma=4.0)

    # 3. Save Output Images
    search_path = os.path.join(pair_dir, "search.png")
    target_path = os.path.join(pair_dir, "target.png")
    json_path = os.path.join(pair_dir, "pair3--groundtruth.json")

    cv2.imwrite(search_path, search_img_uint8)
    cv2.imwrite(target_path, target_img_uint8)

    # 4. Save Groundtruth JSON
    gt_data = {
        "pair_id": "pair3",
        "search_image": "search.png",
        "target_image": "target.png",
        "search_size": [1000, 1000],
        "target_size": [1000, 1000],
        "target_bbox": [gt_x, gt_y, target_w, target_h],
        "target_center": [gt_x + target_w // 2, gt_y + target_h // 2],
        "description": "Design 3: Dense Layout Bottom-Left Crossing Bridge Short Defect (Noise sigma=95)"
    }

    with open(json_path, 'w') as f:
        json.dump(gt_data, f, indent=4)

    print(f"[Pair 3] Generated 1000x1000 search.png (noise sigma=95), 1000x1000 target.png, and pair3--groundtruth.json in {pair_dir}")

if __name__ == "__main__":
    generate_pair3()
