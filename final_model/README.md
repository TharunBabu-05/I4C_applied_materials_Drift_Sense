# Drift-Sense: Final Model Package

Self-contained inference package for the **Siamese Localization System** developed for the I4C Applied Materials Hackathon (Problem Statement 02).

This package contains everything needed to reproduce inference results on a **new machine** — no training required.

---

## 📁 Folder Structure

```
final_model/
├── README.md                            # This file
├── requirements.txt                     # Pinned Python dependencies
├── run_inference.py                     # Main inference & evaluation script
├── best_model_level_resnet4_final.pth   # Trained ResNet checkpoint (~1.4MB)
├── best_model_level_mobilenet_v3.pth    # Trained MobileNetV3 checkpoint (~4.1MB)
├── models/                              # Model architecture definitions
│   ├── __init__.py
│   ├── pyramid_siamese.py               # PyramidSiameseNetwork wrapper
│   └── siamese_encoder.py               # ResNet & MobileNetV3 encoders
└── all_60_pairs/                        # Evaluation dataset (60 SEM image pairs)
    ├── pair_001/
    │   ├── reference.png                # 100x magnification reference
    │   ├── search.png                   # 10x magnification search area
    │   ├── target.png                   # Ground truth crop
    │   └── groundtruth.json             # {center_x, center_y, ...}
    ├── pair_002/
    │   └── ...
    └── pair_060/
        └── ...
```

---

## 🚀 Quick Start (New Laptop Setup)

### Prerequisites
- **Python 3.10+** (tested on 3.10.12)
- **pip** package manager
- **Git** (to clone the repo)

### 1. Clone the Repository
```bash
git clone https://github.com/TharunBabu-05/I4C_applied_materials_Drift_Sense.git
cd I4C_applied_materials_Drift_Sense/final_model
```

### 2. Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

**For GPU (CUDA 12.1):**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4. Run Evaluation on All 60 Pairs
```bash
# Using ResNet encoder (recommended, fastest):
python run_inference.py --evaluate --checkpoint best_model_level_resnet4_final.pth --encoder resnet --verbose

# Using MobileNetV3 encoder:
python run_inference.py --evaluate --checkpoint best_model_level_mobilenet_v3.pth --encoder mobilenet --verbose

# Hybrid mode (NCC + Siamese fusion):
python run_inference.py --evaluate --checkpoint best_model_level_resnet4_final.pth --encoder resnet --mode hybrid --verbose
```

### 5. Single Pair Inference
```bash
python run_inference.py \
    --reference all_60_pairs/pair_001/reference.png \
    --search all_60_pairs/pair_001/search.png \
    --checkpoint best_model_level_resnet4_final.pth \
    --encoder resnet \
    --verbose
```

---

## ⚙️ Command-Line Options

| Argument | Description | Default |
|---|---|---|
| `--mode` | `pure` (sliding window) or `hybrid` (NCC + Siamese) | `pure` |
| `--evaluate` | Run batch evaluation on all 60 pairs | — |
| `--reference` | Path to reference image (100x mag) | — |
| `--search` | Path to search image (10x mag) | — |
| `--checkpoint` | Path to `.pth` model weights | — |
| `--encoder` | `resnet` or `mobilenet` backbone | `resnet` |
| `--verbose` | Print timing and candidate details | `false` |

---

## 🧠 Model Architecture

The system uses a **Pyramid Siamese Network** with shared-weight encoders:

- **ResNet Encoder**: Custom lightweight ResNet with residual blocks → 128-D L2-normalized embedding (~1.4MB)
- **MobileNetV3 Encoder**: MobileNetV3-Small adapted for 1-channel grayscale → 128-D embedding (~4.1MB)

Both encoders are trained with **InfoNCE contrastive loss** on synthetic SEM wafer image pairs.

### Inference Modes

1. **Pure Siamese** (`--mode pure`): Two-phase sliding window (coarse stride-20 → fine stride-1)
2. **Hybrid** (`--mode hybrid`): Fast NCC template matching shortlist → Siamese re-ranking with 0.3×NCC + 0.7×Siamese fusion

---

## 📊 Expected Output

```
======================================================================
  Drift-Sense Evaluation — 60 pairs | Mode: pure
  Device: cpu
======================================================================

  ✓ pair_001: pred=(431,571) gt=(430,570) err=1.4px
  ✓ pair_002: pred=(350,420) gt=(350,420) err=0.0px
  ...

======================================================================
  RESULTS SUMMARY
======================================================================
  Pairs evaluated : 60
  Mean error      : XX.XX px
  Median error    : XX.XX px
  < 50px accuracy : XX/60 (XX.X%)
======================================================================
```
