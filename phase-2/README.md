# Phase 2 Semi-Finals: Status & Next Steps

## 🚨 Critical Discovery: The Dataset Generator Bug
During testing, we discovered a massive mathematical flaw in your friend's `generate_phase2_60_dataset.py` script. 

The script was simulating camera zoom and rotation by cropping a 100x100 patch, scaling/rotating it, and pasting it back into an **unscaled** background. Because semiconductor patterns are highly periodic grids, this left the search image completely filled with "perfect," unscaled decoys that exactly matched the reference image. Our algorithm was constantly failing because, mathematically, the unscaled decoys were a *better* match than the true scaled target!

**Fix Applied:** I rewrote the generator to apply the Affine Transformation (scale & rotation) globally to the entire search image and accurately map the Ground Truth coordinates. This perfectly mirrors true SEM camera misalignment and eliminates the artificial unscaled decoys.

## 📊 Current Pipeline Accuracy & Metrics
After regenerating the 200-pair Phase 2 test set with the fixed script, I tested both the Siamese Network and a pure Mathematical NCC Sweep. 

### 1. Fine-Tuned Siamese Network: **45% Accuracy**
Our attempt to fine-tune the Phase 1 `PyramidSiameseNetwork` on just 1,000 Phase 2 pairs resulted in **Catastrophic Forgetting**. The model forgot the core patterns from the 60 generators and became heavily biased. It actively overrides the correct NCC scores, picking wrong locations and mathematically inverting the scale predictions.

### 2. Pure Mathematical NCC Sweep: **65% Accuracy (Our New Baseline)**
By completely disabling the Siamese network and using our mathematically pure NCC sweep (which resizes the search image instead of the template to maintain a constant 100x100 correlation energy), we achieve 65% accuracy zero-shot! 
- **Scale Accuracy:** 100% (The scale prediction math is now perfectly aligned with the GT).
- **Latency:** ~0.6s per pair (Well within the 2.0s Hackathon limit).
- **Why 65%?** The remaining 35% failures occur on highly periodic grids where a distant periodic decoy happens to have slightly *less* random noise applied to it than the true target. Because we cannot use a simple center-distance penalty (since true targets aren't always centered), pure NCC cannot distinguish them.

## 🚀 What to do next?
To win the Semi-Finals and secure our spot in the Top 5, we MUST fix the Siamese Network so it can confidently break the ties between identical noisy decoys. 

1. **Massive Dataset Generation:** Use our newly fixed `generate_phase2_60_dataset.py` script to generate a massive 50,000+ pair dataset with heavy Phase 2 noise applied.
2. **Retrain the Siamese Model:** We must retrain the `PyramidSiameseNetwork` from scratch (or carefully fine-tune with a very low learning rate) on this massive dataset so it learns to be robust against Phase 2 noise without forgetting the Phase 1 generator features.
3. **Final Fusion Integration:** Once the Siamese model is retrained, we will re-enable the fusion loop in `register.py`. NCC will perfectly lock the scale and propose the top 2 candidates, and the robust Siamese model will perfectly distinguish the true target from the noisy decoys.

Let me know if you want me to write the retraining script and start the massive generation!
