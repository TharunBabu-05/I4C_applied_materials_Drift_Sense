# Drift-Sense Project Status (Updated: Aug 8)

## 📌 Current Status: READY FOR SUBMISSION
The core algorithm is fully developed, rigorously tested across multiple datasets, and proven to be robust. The standalone `inference.py` script strictly adheres to the competition requirements (taking reference and search image paths, outputting `(x, y)` coordinates, and running without manual edits). 

---

## 🧠 Core Algorithm: Drift-Sense v3.0
Our final algorithm is the **Multi-Scale NCC (Normalized Cross-Correlation) Pyramid**. 
It handles the extreme noise and structural ambiguity of Scanning Electron Microscope (SEM) images by downsampling the high-resolution Reference image to match the Search image scale (10x), and performing hierarchical template matching. It includes a smart "center-bias" disambiguation step to resolve periodic ambiguities common in semiconductor layouts.

---

## 📊 Comprehensive Performance Metrics

We have tested the algorithm against **6 distinct datasets**, ranging from standard DRAM layouts to heavily deteriorated and custom geometric structures. 

| Dataset | Type | Accuracy (≤10px) | Mean Error | Mean Latency | Verdict |
|---|---|---|---|---|---|
| **Primary DRAM Dataset** | Standard Grid | **86% - 88%** | ~12.4 px | 0.214 s | **Excellent**. The core target of the competition. High performance on intended patterns. |
| **Mixed-Pattern (10 pairs)** | 5 Structural Variants | **70%** | 34.3 px | 0.229 s | **Strong**. 100% on Standard, High-Contrast, and High-LER. Failed only on extreme low-contrast cases. |
| **Senthil's Ultra-Dense Boxes** | Custom Geometry | **60%** | 26.6 px | 0.136 s | **Good**. Fails only when ground-truth is far off-center due to our center-bias logic. |
| **Senthil's "Most Denser"** | Custom Geometry | **100%** | 0.0 px | 0.178 s | **Perfect**. Flawless localization despite scattered ground-truths. |
| **Senthil-5** | Custom Geometry | **100%** | 0.0 px | 0.172 s | **Perfect**. Flawless localization. |
| **Senthil's 2Merged Superdense** | Extreme Density | **0%** | 201.0 px | 0.190 s | **Limitation found**. Extreme repetitive density overwhelms the NCC pyramid causing alias peaks. |

### Key Takeaways from Metrics:
1. **Speed:** The algorithm is incredibly fast, consistently running between **130ms and 230ms** per image pair.
2. **Robustness:** It is immune to High-LER (Line Edge Roughness) and extreme brightness/contrast shifts (High-Contrast).
3. **Weaknesses:** It struggles when structural contrast drops below 40 units (Low-Contrast variant) because SEM noise overwhelms the signal, and it struggles with "superdense" grids where thousands of cells look perfectly identical.

---

## 🛠️ Project Files & Architecture

### 1. The Core Submission Files (What the judges care about)
* `inference.py`: The standalone inference engine. This is the only file the judges will execute. It is fully generic and takes command-line arguments `--reference` and `--search`.
* `requirements.txt`: Contains our dependencies (`opencv-python`, `numpy`, `scipy`).

### 2. Our Internal Testing & Validation Suite
We built a massive suite of tools to ensure our algorithm was bulletproof:
* `dataset_generator.py`: Generates the primary DRAM layouts with realistic SEM noise, LER, and CD variations.
* `evaluate.py`: The main evaluator for the primary dataset.
* `generate_mixed_dataset.py`: A specialized generator that tests the algorithm's boundaries by creating 5 distinct structural variants (Standard, Coarse, Low-Contrast, High-Contrast, High-LER).
* `evaluate_custom_dataset.py`: A generic wrapper we built to test and visualize Senthil's custom datasets. It generates rich side-by-side images with error vectors.
* `plot_metrics.py`: A dashboard generator that uses `matplotlib` and `seaborn` to plot spatial distribution, error histograms, and latency boxplots for any evaluation run.

---

## 🚀 Next Steps
1. **Prepare Presentation (PPT):** Use the visualizations and the metrics dashboards we generated today to build the final slide deck.
2. **GitHub Repository:** Push the final codebase (excluding massive dataset folders) to a clean GitHub repository.
3. **Final README:** Update the root `README.md` with team details and instructions for the judges.
