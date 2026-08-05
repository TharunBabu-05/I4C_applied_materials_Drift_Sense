# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection

> **I4C Hackathon -- Applied Materials Problem Statement #2**
> Localize a high-resolution reference pattern (100x) inside a noisy,
> lower-resolution search image (10x) for DRAM memory cell arrays.

## Problem Overview

Semiconductor wafer inspection tools revisit the same sites thousands of times
per day, but mechanical drift causes landing errors. DRAM die layouts are
highly periodic (repeating word-line/bit-line grids), so the tool cannot
easily distinguish the correct location from adjacent repeating tiles.

**Our solution:** A 3-level Multi-Scale Normalized Cross-Correlation (NCC)
pyramid that handles the 10x scale difference, independent SEM noise between
captures, and periodic ambiguity -- without any training data or GPU.

---

## Quick Start

### 1. Install dependencies

\\ash
pip install -r requirements.txt
\
### 2. Generate Synthetic Dataset

\\ash
python dataset_generator.py --style DRAM --num_pairs 50 --output_dir ./generated_data --seed 1337
\
### 3. Run Inference on a Single Pair

\\ash
python inference.py --reference generated_data/pair_001/reference.png --search generated_data/pair_001/search.png
\
**Output:** \(x, y)\ -- predicted center of the reference in the search image.

### 4. Run Full Evaluation

\\ash
python evaluate.py --data_dir ./generated_data --output_dir ./results
\
---

## Architecture: DRAM Capacitor-Body Model (v2.5)

We generate **realistic DRAM memory cell array** patterns matching real SEM imagery:

- **Dark square capacitor bodies** -- storage node dielectric (low SE yield = dark)
- **Bright metal interconnect walls** -- tungsten/aluminum grid (high SE yield = bright)
- **Rounded cell corners** from lithography resolution limits
- **Line Edge Roughness (LER)** -- correlated fractal edge perturbation (1-3 px RMS) [6]
- **Critical Dimension (CD) variation** -- global gradient +/-5% across the field
- **Hierarchical block banding** -- subtle brightness bands every ~2000 px (sense-amp rows)
- **4 defect types** -- missing contacts, particle contamination, line bridges (shorts), line breaks (opens)

---

## Algorithm: Multi-Scale NCC Pyramid (v2.5)

\Reference (1000x1000, 100x)
          |
          v
Level 0:  Template 50x50  vs Search 500x500   (coarse: avoids cell aliasing)
          | Top-20 candidates
Level 1:  Template 100x100 vs Search 1000x1000 (nominal: fused score 35%L0+65%L1)
          | Best candidate
Level 2:  Template 200x200 vs 400x400 window   (fine: sub-cell precision)
          | Center-bias disambiguation
          v
     Output: (x, y)
\
**Why Classical (No Deep Learning)?**
- No training data required -- works on unseen test data out-of-the-box
- Fast inference -- ~0.45 s/pair on CPU
- Reproducible -- deterministic, no model weights
- Explainable -- every decision based on mathematical NCC scores

---

## Noise & Augmentation Model

| Augmentation | Model | References |
|-------------|-------|------------|
| Shot noise | Poisson (signal-dependent) | Foi et al. 2008, Joy 2002 |
| Read noise | Gaussian (signal-independent) | Foi et al. 2008, Goldstein 2018 |
| Edge brightening | Sobel + additive blend | Goldstein 2018 |
| Blur | Gaussian PSF | Reimer & Kohl 2008 |
| Beam-current drift | Sinusoidal scan-line modulation | Goldstein 2018 |
| Rotation | Small random angle (+/-0.5 deg) | Goldstein 2018 |
| Vignetting | Radial cos^4 falloff | Goldstein 2018 |
| Intensity drift | Random gain + offset | Postek & Vladar 2013 |
| Line Edge Roughness | Correlated 1D fractal profile | Stoyanov et al. 2018 |
| CD Variation | Global linear gradient +/-5% | Mack 2007, Wong et al. 2020 |

---

## Evaluation Results (v2.5 -- Final)

Evaluated on 50 generated pairs (seed=1337), 5px tolerance:

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **86.0% (43/50)** |
| Median error | 0.00 px |
| Mean inference time | 0.45 s/pair |
| Total evaluation time | 22.5 s |

**Failure Analysis (7 failures -- all noise-induced):**
- 0 periodic ambiguity failures (pyramid fully resolved this challenge)
- 7 noise-induced failures (extreme shot noise in search image overwhelmed correlation)

**Version History:**

| Version | Accuracy | Notes |
|---------|----------|-------|
| v1 | 100% | Artificial crosshair -- trivially easy, not realistic |
| v2 | 62% | Realistic capacitor model, but search noise too heavy |
| v3 | 38% | Over-engineered median/bilateral preprocessing destroyed NCC edges |
| **v2.5** | **86%** | Rollback to proven preprocessing + noise tuning = best result |

---

## File Structure

\|- README.md                    <- You are here
|- requirements.txt             <- Python dependencies
|- dataset_generator.py         <- DRAM synthetic data generator (v2.5)
|- inference.py                 <- Multi-scale NCC localization (v2.5)
|- evaluate.py                  <- Batch evaluation + difficulty grading
|- references.md                <- Academic citations
|- IMPROVEMENT_REPORT.md        <- Analysis and improvement roadmap
|- generated_data/              <- Output of dataset_generator.py
|   |- pair_001/
|   |   |- reference.png        (1000x1000, 100x mag)
|   |   |- search.png           (1000x1000, 10x mag)
|   |   \- ground_truth.json
|   \- metadata.json
\- results/                    <- Output of evaluate.py
    |- evaluation_report.json
    |- success_example.png
    |- failure_example.png
    \- error_distribution.png
\
---

## Tech Stack

- **Language:** Python 3.8+
- **Libraries:** NumPy, Pillow, SciPy, OpenCV, Matplotlib
- **Hardware:** CPU only (no GPU required)

## References

See [eferences.md\](references.md) for the complete list of academic citations.

## Team

- **Team Name:** [Your Team Name]
- **Members:** [Names and Roles]
- **College:** [Your College]

## License

This project was created for the I4C Hackathon 2026.
