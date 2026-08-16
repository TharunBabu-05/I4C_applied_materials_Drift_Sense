import os
import sys
import cv2

def main():
    # Models to evaluate
    models = [
        "best_model_level_resnet4_final.pth",
        "best_model_level_resent3.pth",
        "best_model_level_resent2.pth"
    ]

    out_dir = "model/data_rgb_test"
    
    print("=================================================================")
    print("1. Generating EXACTLY 100 new test samples...")
    print("=================================================================")
    # Running generator for exactly 100 pairs
    os.system(f"python3 model/standalone_dataset_generator/generate_dataset.py --num_pairs 100 --output_dir {out_dir} --seed 4242")

    # Find all generated pairs recursively
    all_pairs = []
    for root, dirs, files in os.walk(out_dir):
        if any(f.startswith("target") for f in files) and any(f.startswith("search") for f in files):
            all_pairs.append(root)

    print(f"\nFound {len(all_pairs)} generated pairs across all splits.")

    print("\n=================================================================")
    print("2. Converting 10 samples to RGB format...")
    print("=================================================================")
    for i in range(min(10, len(all_pairs))):
        pair_path = all_pairs[i]
        for img_name in ["reference.png", "target.png", "search.png"]:
            p = os.path.join(pair_path, img_name)
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(p, img_rgb)
        print(f"  --> Converted {os.path.basename(pair_path)} to RGB (3-channels).")

    print("\n=================================================================")
    print("3. Evaluating the 3 Models on all 100 samples...")
    print("=================================================================")
    
    for ckpt in models:
        print(f"\nEvaluating: {ckpt}")
        if not os.path.exists(ckpt):
            print(f"  [ERROR] Checkpoint not found: {ckpt}")
            continue
            
        # Point to the root directory, evaluate_final will recursively find all 100
        os.system(f"python3 evaluate_final.py --data_dir {out_dir} --checkpoint {ckpt} --encoder resnet")

if __name__ == "__main__":
    main()
