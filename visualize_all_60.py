import os
import sys
import json
import cv2
import math
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.models.pyramid_siamese import PyramidSiameseNetwork

import importlib.util
spec_hybrid = importlib.util.spec_from_file_location("inference_hybrid", "model/inference/inference_hybrid.py")
inference_hybrid = importlib.util.module_from_spec(spec_hybrid)
spec_hybrid.loader.exec_module(inference_hybrid)
localize_hybrid = inference_hybrid.localize_hybrid

def main():
    data_dir = "model/all_60_pairs"
    vis_dir = os.path.join(data_dir, "visualizer")
    os.makedirs(vis_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyramidSiameseNetwork(encoder_type="resnet").to(device)
    model.load_state_dict(torch.load("model/checkpoints/best_model_level1.pth", map_location=device))
    model.eval()

    pairs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "visualizer"]
    
    print("=========================================================")
    print("Generating visualizations for all 60 pairs...")
    print("Finding the 2 Missed images...")
    print("=========================================================\n")
    
    for pair in pairs:
        pair_dir = os.path.join(data_dir, pair)
        files = os.listdir(pair_dir)
        
        ref_name = next((f for f in files if f in ["reference.png", "target.png"]), None)
        search_name = next((f for f in files if f in ["search.png"]), None)
        gt_name = next((f for f in files if "ground" in f.lower() or "gt" in f.lower()), None)
        
        if not ref_name or not search_name or not gt_name:
            continue
            
        ref = os.path.join(pair_dir, ref_name)
        search = os.path.join(pair_dir, search_name)
        gt_path = os.path.join(pair_dir, gt_name)
        
        with open(gt_path, 'r') as f:
            gt = json.load(f)
            if "target_x" in gt:
                gt_x, gt_y = gt["target_x"], gt["target_y"]
            elif "target" in gt and "search_center_xy" in gt["target"]:
                gt_x, gt_y = gt["target"]["search_center_xy"]
            else:
                gt_x, gt_y = float(gt.get('center_x', 500)), float(gt.get('center_y', 500))
                
        pred_x, pred_y = localize_hybrid(model, ref, search, device, verbose=False)
        
        err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        
        # Visualize
        search_img = cv2.imread(search, cv2.IMREAD_GRAYSCALE)
        search_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        
        # Ground Truth: Green (100x100)
        cv2.rectangle(search_color, (int(gt_x) - 50, int(gt_y) - 50), (int(gt_x) + 50, int(gt_y) + 50), (0, 255, 0), 3)
        
        # Prediction: Red (100x100)
        cv2.rectangle(search_color, (int(pred_x) - 50, int(pred_y) - 50), (int(pred_x) + 50, int(pred_y) + 50), (0, 0, 255), 3)
        
        # Text Information
        cv2.putText(search_color, f"GT: ({gt_x}, {gt_y})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(search_color, f"Pred: ({pred_x}, {pred_y})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(search_color, f"Error: {err:.2f} px", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        
        # Draw a line connecting the centers if it's a miss
        if err > 5.0:
            cv2.line(search_color, (int(gt_x), int(gt_y)), (int(pred_x), int(pred_y)), (0, 165, 255), 3)
        
        out_path = os.path.join(vis_dir, f"{pair}.png")
        cv2.imwrite(out_path, search_color)
        
        if err > 5.0:
            print(f" [MISSED] {pair} | Error: {err:.2f}px | GT: ({gt_x}, {gt_y}) | Pred: ({pred_x}, {pred_y})")

    print(f"\n=========================================================")
    print(f"Visualization complete! Check the folder:")
    print(f"  {vis_dir}")
    print("=========================================================")

if __name__ == "__main__":
    main()
