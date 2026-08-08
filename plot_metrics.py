#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_metrics_dashboard(report_path, output_path):
    with open(report_path, 'r') as f:
        data = json.load(f)

    pairs = data['pairs']
    
    gt_x = [p['gt_x'] for p in pairs]
    gt_y = [p['gt_y'] for p in pairs]
    pred_x = [p['pred_x'] for p in pairs]
    pred_y = [p['pred_y'] for p in pairs]
    errors = [p['error_px'] for p in pairs]
    latencies = [p['inf_time'] * 1000 for p in pairs] # convert to ms
    statuses = [p['passed'] for p in pairs]

    # Metrics
    acc = data['accuracy_pct']
    mean_err = data['mean_error_px']
    max_err = max(errors)
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)

    # Set up the dashboard figure
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Performance Dashboard: {data['dataset']}", fontsize=20, fontweight='bold', y=0.98)

    # 1. Scatter Plot (GT vs Pred)
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(gt_x, gt_y, c='green', s=100, label='Ground Truth', marker='P', alpha=0.7)
    ax1.scatter(pred_x, pred_y, c='red', s=50, label='Prediction', marker='o', alpha=0.7)
    
    # Draw error lines
    for i in range(len(pairs)):
        ax1.plot([gt_x[i], pred_x[i]], [gt_y[i], pred_y[i]], 'y-', alpha=0.5)

    ax1.set_title("Spatial Distribution: Ground Truth vs Prediction", fontsize=14)
    ax1.set_xlabel("X Coordinate (pixels)")
    ax1.set_ylabel("Y Coordinate (pixels)")
    ax1.legend()
    ax1.invert_yaxis() # Image coordinates (y goes down)

    # 2. Error Distribution
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(errors, bins=10, kde=True, color='crimson', ax=ax2)
    ax2.axvline(data['tolerance_px'], color='gold', linestyle='--', label=f"Tolerance ({data['tolerance_px']}px)")
    ax2.set_title("Localization Error Distribution", fontsize=14)
    ax2.set_xlabel("Error Magnitude (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.legend()

    # 3. Latency Distribution
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(x=latencies, color='royalblue', ax=ax3)
    sns.stripplot(x=latencies, color='black', alpha=0.5, ax=ax3)
    ax3.set_title("Inference Latency Distribution", fontsize=14)
    ax3.set_xlabel("Latency (ms)")

    # 4. Metrics Summary Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    metrics_text = (
        f"Summary Statistics\n"
        f"--------------------------\n"
        f"Total Pairs Tested : {len(pairs)}\n"
        f"Overall Accuracy   : {acc:.1f}%\n"
        f"Tolerance Threshold: {data['tolerance_px']} px\n\n"
        f"Localization Error:\n"
        f"  Mean Error       : {mean_err:.2f} px\n"
        f"  Max Error        : {max_err:.2f} px\n\n"
        f"Inference Latency:\n"
        f"  Mean Latency     : {mean_lat:.1f} ms\n"
        f"  95th Percentile  : {p95_lat:.1f} ms\n"
    )
    
    ax4.text(0.1, 0.5, metrics_text, fontsize=16, fontfamily='monospace', 
             verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_json")
    parser.add_argument("output_png")
    args = parser.parse_args()
    generate_metrics_dashboard(args.report_json, args.output_png)
