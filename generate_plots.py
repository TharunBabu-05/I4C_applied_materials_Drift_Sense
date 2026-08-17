import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_plots(csv_path="results_manifest.csv", output_dir="model/all_60_pairs/visualizer"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run visualize_all_60.py first.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Read CSV
    gt_x, gt_y = [], []
    pred_x, pred_y = [], []
    errors = []
    times = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_x.append(float(row['gt_x']))
            gt_y.append(float(row['gt_y']))
            pred_x.append(float(row['pred_x']))
            pred_y.append(float(row['pred_y']))
            errors.append(float(row['error_px']))
            times.append(float(row['inference_time_ms']))
            
    gt_x = np.array(gt_x)
    gt_y = np.array(gt_y)
    pred_x = np.array(pred_x)
    pred_y = np.array(pred_y)
    errors = np.array(errors)
    times = np.array(times)
    
    print("Generating statistical plots for the Hackathon Results...")
    
    # Set the style manually for base matplotlib
    plt.style.use('dark_background')
    
    # ---------------------------------------------------------
    # Plot 1: Error Distribution Histogram (Zoomed)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # 1-pixel bins from 0 to 30 to show EXACTLY how many are 0px, 1px, 2px, etc.
    bins = np.arange(0, 31, 1.0)
    
    plt.hist(errors, bins=bins, color='cyan', alpha=0.7, edgecolor='white')
    plt.axvline(x=5.0, color='red', linestyle='--', label='5px Tolerance Threshold', linewidth=2)
    
    plt.title('Distribution of Localization Error (0-30px)', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Error (Pixels)', fontsize=12, color='white')
    plt.ylabel('Frequency (Number of Images)', fontsize=12, color='white')
    
    outliers = sum(e > 30.0 for e in errors)
    if outliers > 0:
        plt.text(0.95, 0.5, f"Outliers > 30px: {outliers} images", 
                 horizontalalignment='right', 
                 transform=plt.gca().transAxes,
                 color='yellow', fontsize=12, fontweight='bold')
                 
    plt.xlim(0, 30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_error_distribution.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 2: Scatter of Ground Truth vs Predicted Centers
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    
    # Plot GT
    plt.scatter(gt_x, gt_y, c='lime', label='Ground Truth', alpha=0.6, s=100, marker='o')
    # Plot Predictions
    plt.scatter(pred_x, pred_y, c='magenta', label='Predicted', alpha=0.6, s=50, marker='x')
    
    # Draw lines connecting errors
    for i in range(len(errors)):
        if errors[i] > 5.0:
            plt.plot([gt_x[i], pred_x[i]], [gt_y[i], pred_y[i]], 'r-', alpha=0.5)
            
    plt.title('Spatial Distribution: GT vs Predicted Coordinates', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('X Coordinate (Pixels)', fontsize=12, color='white')
    plt.ylabel('Y Coordinate (Pixels)', fontsize=12, color='white')
    
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_spatial_scatter.png'), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 3: Inference Time vs Error
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(times, errors, c=errors, cmap='coolwarm', s=100, alpha=0.8, edgecolors='w')
    
    plt.axhline(y=5.0, color='red', linestyle='--', label='5px Tolerance Threshold', linewidth=2)
    
    plt.title('Robustness: Inference Time vs Localization Error', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Inference Time (ms)', fontsize=12, color='white')
    plt.ylabel('Localization Error (Pixels)', fontsize=12, color='white')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error (px)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_inference_robustness.png'), dpi=300)
    plt.close()
    
    print(f"Successfully generated 3 high-quality plots in: {output_dir}")
    print(" - plot_error_distribution.png")
    print(" - plot_spatial_scatter.png")
    print(" - plot_inference_robustness.png")

if __name__ == "__main__":
    generate_plots()
