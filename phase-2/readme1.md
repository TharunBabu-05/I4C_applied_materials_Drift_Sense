# 🏆 Phase 2 — Applied Materials I4C Hackathon (Semi-Finals)
> **Status: TOP 15 — Semi-Finalist | Target: Top 5 → Delhi Finals**  
> **Last Updated: August 29, 2026**

## 📋 Phase 2 Overview

Phase 2 is a **blind evaluation round** where the organizers run their own 200-pair secret test set against our submitted `register.py` script. We never see the images — we only see our scores.

### Key Differences from Phase 1:

| Feature | Phase 1 | Phase 2 |
|---|---|---|
| Task | Localize (x, y) only | Localize (x, y) + Pose (θ, scale) + Rejection |
| Output | `(x, y)` printed | `predictions.csv` with full contract |
| Scale | Fixed 10× | Variable **[8×, 12×]** |
| Rotation | 0° | **±5°** |
| Absent targets | No | **Yes! Set C (40 pairs)** |
| Image types | Grayscale SEM | Grayscale + **RGB Optical (Set D)** |
| Dataset size | Custom | **200 blind pairs (organizer-generated)** |

---

## 📦 The 200 Blind Pairs — Dataset Structure

| Set | Pairs | Type | Description | Feeds |
|---|---|---|---|---|
| **Set A** | 70 | Nominal Pose | Standard noise, scale [8,12]×, ±5° rotation | Localization + Pose |
| **Set B** | 70 | Degraded | Charging, scan distortion, defocus, elevated shot noise, polygon scaling ±20% in **4 severity levels** | Localization + Pose |
| **Set C** | 40 | Absent | No true instance — a different die region. Correct answer: `found = 0` | Rejection F1 |
| **Set D** | 20 | Optical (Bonus) | RGB 3-channel optical-microscope analogue, reference present | +6 Bonus pts |

---

## 📤 Output Contract — `register.py`

### SINGLE ENTRY POINT (mandatory signature):
```bash
python register.py --input pairs.csv --output predictions.csv
```

### `predictions.csv` — ONE ROW PER PAIR:

| Column | Type | Description |
|---|---|---|
| `pair_id` | str | Exactly as supplied in `pairs.csv` |
| `x` | float | Match centre in wide-search coords, subpixel allowed |
| `y` | float | Match centre in wide-search coords, subpixel allowed |
| `theta` | float | Rotation in degrees, CCW positive, about the match centre |
| `scale` | float | Recovered down-scaling factor, nominally in [8, 12] |
| `found` | int | 1 or 0. When 0, write 0 in the pose columns |
| `score` | float | Your own confidence, any monotonic scale |

> ⚠️ **CRITICAL: Every `pair_id` must appear exactly once. A missing row scores zero.**

### Reference Machine Constraints:
- 4-core x86 CPU, **8 GB RAM**
- **No GPU, no network**
- Python 3.11
- Weights ship inside the zip — nothing downloads at runtime
- **Median ≤ 5 s per pair | Hard timeout = 20 s → scores zero**

### Also required inside the submission zip:
- `requirements.txt` (from pip freeze)
- `generate_dataset.py` (documented)
- `failure_analysis.pdf` (max 2 pages)

---

## 🏅 Scoring — 100 Points + 10 Bonus

| Points | Component | Description |
|---|---|---|
| **40** | Localization | Sets A+B. Tiered: ≤1px=1.00, ≤2px=0.80, ≤3px=0.60, ≤5px=0.40, >5px=0.00. Weighted: 0.45·A + 0.55·B |
| **20** | Pose Recovery | Scale (10 pts) + Rotation (10 pts). Only scored where localization credit > 0 |
| **15** | Rejection | F1 on the `found` flag across all 180 grayscale pairs. Never rejecting = 0 pts |
| **10** | Confidence Calibration | AUC of your `score` column against per-pair correctness |
| **5** | Efficiency | Relative quartile ranking on median wall-clock per pair |
| **10** | Generator + Citations + Failure Analysis | Carried from Phase 1, re-judged under Phase 2 conditions |
| **+10 Bonus** | Set D optical (≥0.40) + Sets A–C (≥0.50) + Rejection F1 ≥ 0.90 | |

### Credit Tiers for Localization:
- ≤ 1 px → **1.00 credit**
- ≤ 2 px → **0.80 credit**
- ≤ 3 px → **0.60 credit**
- ≤ 5 px → **0.40 credit**
- > 5 px → **0.00 credit**

### Pose Recovery Tiers:
| Credit | Scale Error | Rotation Error |
|---|---|---|
| 1.00 | ≤ 1% | ≤ 0.25° |
| 0.60 | ≤ 2% | ≤ 0.5° |
| 0.30 | ≤ 5% | ≤ 1.0° |

---

## ✅ What Is Allowed / ❌ What Disqualifies

### ✅ ALLOWED:
- Extending Phase 1 method to search over scale [8,12] and rotation ±5°
- Hard-coding the disclosed bounds [8,12] and ±5°
- Regenerating dataset with scale and rotation augmentation
- Data augmentation, retraining, hyperparameter and threshold changes
- Classical, learned, or hybrid localization

### ❌ DISQUALIFIES (NO APPEAL):
- Any **network access** during the scored run
- **Hard-coding filenames** or reading outside supplied paths
- A method materially different from Phase 1 declared approach
- Proprietary fab or non-public layout data in the generator
- Mixing organizer test data into training

---

## 📅 Timeline

| Milestone | Day |
|---|---|
| Addendum released | T+0 |
| Sample pairs published | T+2 |
| Questions close (I/O contract clarifications) | T+3 |
| **Submission due (23:59, code frozen)** | **T+7** |
| Scored run by organizers | T+8-9 |
| **Top 10 announced, Phase 3 begins** | **T+10-11** |

---

## 🛠️ Phase 2 Generator — `phase2_generator_60/`

Your friend (Senthil) has built a solid **60-architecture generator** that covers all 4 Phase 2 sets. Here is its current status:

### ✅ What's Working:
- **60 Round-1 architecture scripts** in `clean_60_scripts/`
- Generates all 4 sets: A (nominal), B (degraded, 4 severity levels), C (absent targets), D (RGB optical)
- Scale range [8×, 12×] ✅
- Rotation range ±5° ✅
- Full canvas spread: targets at [200, 800] ✅
- Absent pair logic for Set C (uses a different arch for reference) ✅
- RGB optical conversion for Set D ✅
- Manifest CSV output ✅

### ⚠️ What Needs Attention (Gaps vs. Requirements):
1. **`register.py` doesn't exist yet** — This is the single most critical file. It is the entry point the judges will run. **WE MUST BUILD THIS.**
2. **`theta` and `scale` estimation is missing in `inference.py`** — Phase 1 only predicted `(x, y)`. Phase 2 needs `theta` and `scale` too.
3. **Rejection logic (`found = 0`) is missing** — We need a threshold/classifier to detect Set C absent pairs.
4. **`confidence score`** — We need a meaningful `score` value per pair for calibration credit.
5. **Set D (RGB) handling** — Current inference is grayscale-only.
6. **`failure_analysis.pdf`** — Required in the submission zip. Max 2 pages.

---

## 🚀 What To Do Next (Prioritized Action Plan)

### CRITICAL (Must-Have for Submission):
1. **Build `register.py`** — The main entry point script that reads `pairs.csv` and outputs `predictions.csv`.
2. **Add Scale Estimation** — After localizing, compute the scale from the NCC peak width or use pyramid search over scale [8, 12].
3. **Add Rotation Estimation** — Extend NCC search to include rotation search ±5° in steps of 0.25°.
4. **Build Rejection Classifier** — Use NCC peak confidence to detect absent pairs (Set C). If max NCC score < threshold → `found = 0`.
5. **Generate training data** with `generate_phase2_60_dataset.py` and retrain/tune.

### HIGH PRIORITY:
6. **Add RGB/Grayscale conversion** — In `register.py`, auto-detect if search image is RGB, convert to grayscale.
7. **Tune rejection threshold** using Set C generated pairs.
8. **Compute confidence score** as a normalized NCC peak score.

### DOCUMENTATION (Required for Full Score):
9. **Write `failure_analysis.pdf`** — 2-page analysis of where the algorithm fails and why.
10. **Update `requirements.txt`** with `pip freeze`.

---

## 📂 Folder Structure

```
phase-2/
├── requrements_for_phase_2/     # Phase 2 spec slides (images)
│   ├── image.png                # Dataset structure (Sets A-D)
│   ├── image2.png               # Output contract + run environment
│   ├── image3.png               # Timeline
│   ├── image4.png               # Allowed / disqualified
│   ├── image5.png               # Credit tiers (localization + pose)
│   └── image6.png               # Scoring breakdown (100 + 10 pts)
│
├── phase2_generator_60/         # Your friend's dataset generator
│   ├── clean_60_scripts/        # 60 semiconductor arch generators
│   ├── degradation_engine.py    # 17 SEM degradation models
│   ├── noise_engine.py          # Stochastic noise primitives
│   ├── generate_phase2_60_dataset.py  # Main generator (200 pairs: A+B+C+D)
│   ├── master_generator_v2.py   # Core pattern + ground truth mapper
│   ├── validate_phase2_dataset.py     # Dataset validation tool
│   ├── INSTRUCTIONS.txt         # Usage instructions
│   └── README.md                # Quick-start guide
│
└── README.md                    # THIS FILE
```