import sys, argparse, time, os
import cv2

# Ensure we can import from the phase-2 directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from register import run_hybrid_inference

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Master Inference Script")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    parser.add_argument("--checkpoint", required=False, default=None, help="Path to model checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()

    ref_path = args.reference
    search_path = args.search
    chk = args.checkpoint

    if not os.path.exists(ref_path):
        print(f"Error: Reference image {ref_path} not found.")
        sys.exit(1)
        
    if not os.path.exists(search_path):
        print(f"Error: Search image {search_path} not found.")
        sys.exit(1)

    t0 = time.time()
    
    # Run the hybrid Phase 2 inference
    x, y, theta, scale, found, score = run_hybrid_inference(ref_path, search_path, chk)
    
    t1 = time.time()
    
    if args.verbose:
        print(f"Inference completed in {(t1 - t0)*1000:.1f}ms")
        
    if found:
        print(f"{x:.1f}, {y:.1f}, {scale:.3f}, {theta:.3f}")
    else:
        print(f"0.0, 0.0, 10.0, 0.0")

if __name__ == "__main__":
    main()
