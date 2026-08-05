# 🎯 I4C Hackathon — Problem Statement #2: Drift-Sense Analysis

## The Problem in Plain English

**Applied Materials** (a semiconductor equipment giant) has a real-world problem: their wafer inspection tools need to revisit the **exact same spot** on a silicon wafer thousands of times a day. But the mechanical motion stages drift over time due to thermal expansion, vibration, and mechanical slack. When the tool revisits a location, it lands a few pixels off — and because all dies on a wafer look **nearly identical** (repeating circuit patterns), the tool can't easily tell it's in the wrong spot.

> [!IMPORTANT]
> Your job: Build an **AI-powered algorithm** that, given a high-resolution "reference" image of a target site and a lower-resolution "wide search" image, can **find exactly where the reference pattern appears** in the search image and return its **(x, y) center coordinates**.

---

## 🔬 Your Choice: DRAM-Style Architecture

Since you chose **DRAM-style**, your synthetic images must mimic:

| Feature | Description |
|---------|-------------|
| **Word lines** | Periodic **horizontal** lines |
| **Bit lines** | Periodic **vertical** lines crossing word lines at right angles |
| **Contact/via dots** | Small bright dots at **every intersection** of word and bit lines |
| **Overall look** | High-contrast, fine pitch, **extremely regular** grid pattern |

This is what makes DRAM hard: the grid is so periodic that hundreds of locations look nearly identical — classical template matching produces tons of false positives.

---

## 📐 The Image Pair Relationship

```
┌─────────────────────────────────────┐
│  REFERENCE IMAGE (100x zoom)        │
│  • 1000 × 1000 pixels               │
│  • 1 nm/pixel → 1 µm × 1 µm FOV    │
│  • Sharp, less noisy                 │
│  • High-resolution capture           │
└─────────────────────────────────────┘
          ↓ Shrunk 10x ↓
┌─────────────────────────────────────┐
│  SEARCH IMAGE (10x zoom)            │
│  • 1000 × 1000 pixels               │
│  • 10 nm/pixel → 10 µm × 10 µm FOV │
│  • Noisier, lower resolution         │
│  • Reference appears as ~100×100 px  │
│    patch somewhere inside            │
└─────────────────────────────────────┘
```

> [!NOTE]
> The reference pattern occupies **~1% of the search image area** (100×100 pixels inside 1000×1000). Finding it is like finding a needle in a haystack — especially when every part of the haystack looks almost the same.

---

## ✅ Exactly What You Must Build (3 Major Deliverables)

### Deliverable 1: Synthetic DRAM Dataset Generator (30% of marks)

A **standalone Python script** (`dataset_generator.py`) that:

- [x] Generates **DRAM-style** image pairs (Reference + Search)
- [x] Accepts parameters: architecture style, number of pairs, output directory
- [x] Creates a **1000×1000 reference** image with periodic word-line/bit-line grid + contact dots
- [x] Creates a **1000×1000 search** image by tiling a larger DRAM layout and downsampling (10x relationship)
- [x] Places the reference pattern at a **random** known location in the search image
- [x] Records **ground truth (x, y)** center coordinates for each pair
- [x] Adds **independent sensor noise** to each image (NOT the same noise on both!)
- [x] Applies **edge-brightening** to mimic real SEM imaging behavior
- [x] Includes realistic degradations: **blur, rotation, scaling variations**
- [x] Search image must be **more noisy** than reference image
- [x] Generates **minimum 30 randomized pairs**

> [!WARNING]
> **Citation Requirement**: Every augmentation choice, noise model, and structural parameter MUST be justified with **2-3 credible public references** (academic papers, textbooks, patents on semiconductor structure or SEM imaging). This is heavily weighted.

### Deliverable 2: Localization / Inference Algorithm (50% of marks)

A **standalone Python script** (`inference.py`) that:

- Takes **two inputs**: path to reference image + path to search image
- Outputs **one (x, y) coordinate** — the predicted center of the reference pattern in the search image
- If multiple matches found → return the one **closest to the center** of the search image
- Must run **without manual edits** — Applied Materials will run it directly on their test data
- Can use **classical ML or deep learning** (your choice)
- Must handle the **10x scale difference** correctly

> [!CAUTION]
> **This is the MOST CRITICAL file.** Applied Materials will run it directly on their secret test data. If it doesn't run, you **cannot be scored**. Test it on a fresh machine before submitting.

### Deliverable 3: PPT Presentation (Using i4C Template)

| Slide | Content Required |
|-------|-----------------|
| **1** | Team name, member names, roles, college, contact |
| **2** | Why navigation-error recovery matters in semiconductor wafer inspection |
| **3** | Your approach: DRAM-style, chosen algorithm (classical ML vs DL), why it's better than template matching |
| **4** | Detailed solution: dataset generator design, noise models, augmentation, localization algorithm, pipeline diagram, **citations** |
| **5** | Innovation & uniqueness: what makes your approach different? |
| **6** | Results: accuracy on 30+ test cases, computation time, one SUCCESS example, one HONEST FAILURE example |
| **7** | Tech stack, hardware used, inference time, model size |
| **8** | GitHub link (mandatory), video demo link (optional) |
| **9** | All references/citations |

---

## 📊 Scoring Breakdown

| Weight | Category | What They're Looking For |
|--------|----------|------------------------|
| **50%** | **Inference accuracy** | Correct (x, y) predictions on their secret test data + computation time |
| **30%** | **Augmentation quality** | How realistic your synthetic DRAM SEM images look, backed by literature citations |
| **10%** | **Failure analysis** | Root cause explanation + explainability on cases where your algorithm fails |
| **Bonus** | **RGB generalization** | If your solution also works on optical microscope (RGB, 3-channel) images |

---

## 📦 GitHub Repository Structure (Mandatory)

```
your-repo/
├── README.md                    # Complete setup instructions (clone → run)
├── requirements.txt             # pip freeze output
├── dataset_generator.py         # Standalone dataset generator script
├── inference.py                 # Standalone localization script (THE KEY FILE)
├── train.py / train.ipynb       # Training script (if DL method used)
├── model_weights/               # .pt / .h5 / .onnx files (if DL method used)
└── references/                  # Citation documents (PDF or markdown)
```

---

## 🧠 Key Technical Challenges to Solve

1. **High Periodicity Ambiguity** — DRAM grids repeat with sub-pixel variation. Your algorithm must distinguish the *correct* tile from hundreds of near-identical tiles.

2. **10x Scale Mismatch** — The reference is at 100x zoom; the search is at 10x. You need a robust multi-scale matching strategy.

3. **Independent Noise** — Reference is sharper; search image is noisier. Your algorithm must be noise-robust.

4. **SEM-Realistic Synthesis** — Edge-brightening, Poisson/Gaussian noise, realistic line widths/spacings based on literature.

5. **Speed** — Computation time is part of the score. Your inference should be fast on a 1000×1000 image.

---

## 🔑 Summary: What Applied Materials Expects From You

> [!IMPORTANT]
> In one sentence: **Build a Python pipeline that generates realistic synthetic DRAM SEM image pairs, then finds a small high-resolution reference pattern (shrunk 10x) inside a large noisy search image, returning its exact (x, y) center — with every design choice backed by academic citations, and with honest analysis of where it fails.**

They want to see:
- **Engineering rigor** (citations, realistic synthesis)
- **Algorithm robustness** (handles noise, periodicity, scale)
- **Honest self-evaluation** (failure cases + root cause)
- **Reproducible code** (runs on a fresh machine without modification)
