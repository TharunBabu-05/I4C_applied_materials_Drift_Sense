# Drift-Sense Siamese Multi-Scale Architecture Proposal

This document outlines the proposed architecture for converting the existing Drift-Sense classical NCC inference pipeline into a trainable Siamese neural network, while preserving the core 3-level coarse-to-fine localization logic.

## 1. Technical Appropriateness of Siamese + 3-Level Architecture

The proposed Siamese + 3-level architecture is **highly appropriate and technically sound** for this problem:
- **Why Siamese:** The task is fundamentally a template matching / correspondence problem (finding a 100x magnification reference pattern within a noisy 10x magnification search image). Siamese networks excel at learning similarity metrics between image patches robust to noise, lighting variations, and minor distortions, without needing to classify absolute object categories.
- **Why 3-Level Pyramid:** The 10x scale difference means the reference pattern occupies only ~1% of the search image. Attempting to run a dense neural feature extractor across the full 1000x1000 search image at high resolution is computationally prohibitive. The 3-level pyramid gracefully solves this:
  - **Level 0 (Coarse):** Quickly eliminates 99% of the search space, focusing only on regions with matching global structures. It avoids cell-level aliasing (periodic ambiguity) by operating at a low frequency.
  - **Level 1 (Nominal):** Verifies the candidates using medium-scale structural details.
  - **Level 2 (Fine):** Achieves sub-pixel/sub-cell accuracy on a tiny cropped window, keeping the computational burden minimal where high resolution is required.

## 2. Proposed Architecture Diagram

```text
                DRIFT-SENSE SIAMESE MULTI-SCALE NETWORK
                       │
                       ▼
              SEM Reference Image (1000x1000)
                       │
                       ▼
             Standard Preprocessing (Eq. Hist + Light Blur)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Level 0         Level 1         Level 2
     Coarse          Nominal           Fine
    (50x50)        (100x100)        (200x200)
        │              │              │
        └──────────────┼──────────────┘
                       ▼
            Shared Siamese Encoder (CNN)
                 (Lightweight: e.g., MobileNetV2 or Custom ResNet)
                       │
                       ▼
             128-D Feature Embeddings
                       │
                       ▼
            Cosine Similarity Head
                       │
                       ▼
               Candidate Ranking (Top-K)
                       │
                       ▼
                Fine Refinement (Level 2)
                 (Delta Coordinate Prediction / Dense Score Map)
                       │
                       ▼
              Center-Bias Physical Prior (Disambiguation)
                       │
                       ▼
                 Final (x,y)
```

## 3. Tensor and Data Flow

### Level 0 (Coarse Search)
- **Reference Input:** Downscaled to 50x50 (Tensor: `[1, 1, 50, 50]`)
- **Search Input:** Downscaled to 500x500 (Tensor: `[1, 1, 500, 500]`)
- **Flow:** 
  1. The search image is divided into overlapping candidate patches of 50x50 (implemented efficiently via fully convolutional layers or `F.unfold`). 
  2. The shared encoder extracts a 128-D embedding for the reference and all search patches.
  3. The similarity head computes scores.
  4. Output: A coarse 2D similarity map. We extract the Top-20 (x, y) coordinates.

### Level 1 (Nominal Verification)
- **Reference Input:** Downscaled to 100x100 (Tensor: `[1, 1, 100, 100]`)
- **Search Input:** 1000x1000. Instead of full convolution, we crop 100x100 windows around the Top-20 candidates from Level 0 (Tensor: `[20, 1, 100, 100]`).
- **Flow:**
  1. The encoder processes the reference and the 20 candidate windows.
  2. Similarity is computed.
  3. Scores from Level 0 and Level 1 are fused using a trainable fusion head (or fixed 0.35/0.65 weights as a baseline).
  4. Output: The absolute best candidate (or Top-5 for disambiguation).

### Level 2 (Fine Refinement)
- **Reference Input:** Downscaled to 200x200 (Tensor: `[1, 1, 200, 200]`)
- **Search Input:** A 200x200 crop from the 1000x1000 search image around the best Level-1 candidate, upscaled 2x to 400x400 (Tensor: `[1, 1, 400, 400]`).
- **Flow:**
  1. The encoder computes dense feature maps for the reference and search window.
  2. A localized similarity map is generated. 
  3. Instead of direct coordinate regression (which is often brittle), we predict spatial offsets `Δx, Δy` via a small localized head or by finding the sub-pixel peak of the localized similarity map (using spatial soft-argmax).
  4. Output: `Δx, Δy` offsets relative to the Level-1 center.

## 4. Recommended Siamese Loss

**Recommendation:** A combination of **Margin-based Contrastive Loss** and a **Hard Negative Mining** strategy.

- **Primary Loss:** `Contrastive Loss = Y * D^2 + (1-Y) * max(margin - D, 0)^2`
  Where `D` is the L2 distance or `1 - CosineSimilarity`, and `Y` is the label (1 for match, 0 for mismatch).
- **Why not Triplet Loss?** Triplet loss requires careful batch mining (anchor, positive, negative), which can be unstable. Contrastive loss with explicit hard-negative sampling per batch is easier to control and tune for periodic structures.
- **Alternative:** InfoNCE (Normalized Temperature-scaled Cross Entropy) loss if we use large batches, as it naturally handles multiple negatives.

## 5. Recommended Encoder Architecture

**Recommendation:** A **Custom Lightweight ResNet-like Architecture**.
Do not use a heavy ResNet-50 or ViT. We need fast CPU inference.
- **Input:** 1-channel Grayscale
- **Structure:**
  - `Conv2d 3x3 (1 -> 16), BatchNorm, ReLU`
  - `MaxPool2d 2x2`
  - `Residual Block (16 -> 32 channels)`
  - `MaxPool2d 2x2`
  - `Residual Block (32 -> 64 channels)`
  - `MaxPool2d 2x2`
  - `Residual Block (64 -> 128 channels)`
  - `AdaptiveAvgPool2d (1x1)` or `Flatten + Linear` depending on whether we need spatial preservation for Level 2.
- **Output Embedding:** 128-D vector.
- **Why:** 128-D is sufficient for encoding structural layouts without overfitting to specific noise instances. It's extremely fast on CPU.

## 6. Dataset Generation Strategy

We will build upon the existing `dataset_generator.py`. The generation script will explicitly generate `(Reference, Positive Candidate, Hard Negative Candidate, Easy Negative Candidate)` tuples.
- **Process:** Generate a large master DRAM layout. Extract the Reference crop. 
- **Scale:** Apply the 10x downscale.
- **Variations:** Apply realistic independent SEM noise, LER, CD variation, and structural defects (missing contacts, bridges) to the master before extracting pairs.

## 7. Positive/Negative/Hard-Negative Pair Strategy

To combat periodic ambiguity, the training data must force the network to look at structural uniqueness, not just general DRAM brightness.
- **Positive Pair:** Reference + Crop exactly at the target location (with independent noise applied).
- **Easy Negative Pair:** Reference + Crop from a random location far from the target or a completely empty/different structure.
- **Hard Negative Pair (Crucial):** Reference + Crop centered exactly 1 cell pitch away (e.g., +45px, +45px). This forces the network to learn the subtle differences (defects, LER, CD variations) that distinguish two neighboring periodic cells.

## 8. Noise and Defect Model

We will preserve the highly successful v2.5 noise model parameters:
- **Noise:** Independent Poisson (shot noise) and Gaussian (read noise) with higher intensity for the 10x search image.
- **Defects:** Missing contacts, particle contamination, line bridges, and line breaks.
- **Structural:** Line Edge Roughness (LER) with fractal correlation, Critical Dimension (CD) gradients.
- **SEM Physics:** Edge brightening (topographic contrast) and slow sinusoidal beam-current drift.

## 9. Training Hyperparameters

- **Optimizer:** AdamW (helps with weight regularization to prevent overfitting on synthetic data).
- **Learning Rate:** 1e-3, stepping down to 1e-4 and 1e-5 via CosineAnnealingLR.
- **Batch Size:** 32 or 64 (pairs or triplets).
- **Epochs:** 50-100 (synthetic data allows infinite generation, so we define an epoch as ~10,000 generated pairs).
- **Mixed Precision:** FP16 (if GPU available during training) for speed.

## 10. Evaluation Metrics

During evaluation, we will compute:
1. **Mean/Median Euclidean Error:** `sqrt((pred_x - true_x)^2 + (pred_y - true_y)^2)`.
2. **Accuracy @ Tolerance:** % of predictions within 1px, 2px, 5px, 10px.
3. **Inference Time (CPU/GPU):** milliseconds per pair.
4. **Top-K Candidate Recall:** % of times the true location is within the Level-0 and Level-1 candidate lists.

## 11. Advantages and Disadvantages compared to NCC

**Advantages:**
- **Noise Robustness:** The neural encoder learns to completely ignore shot/read noise, whereas NCC can be tricked by random noise spikes.
- **Defect Tolerance:** The network can learn that a "missing contact" doesn't change the underlying periodic structure, while NCC suffers a harsh penalty for misaligned pixels.
- **Semantic Understanding:** Siamese networks compare high-level features rather than raw pixel intensities, handling contrast/brightness variations more natively than normalized correlation.

**Disadvantages:**
- **Training Time:** Requires data generation and GPU training time.
- **Explainability:** NCC gives an exact mathematical correlation score. A neural network is a black box.
- **Out-of-Distribution Degradation:** If Applied Materials' real test data contains architectures vastly different from our generated synthetic data (e.g., highly skewed FinFETs when trained only on DRAM), NCC might generalize better while the CNN might fail completely.

## 12. Potential Failure Modes

- **Overfitting to Synthetic Artifacts:** The network might learn to match the specific pseudorandom LER patterns generated by our algorithm rather than general DRAM structures.
- **Periodic Collapse:** If hard negatives aren't sampled correctly, the network might predict high similarity for *all* DRAM cells, rendering the score map useless.
- **Level-0 Misses:** If the coarse Level-0 encoder is too aggressive, it might discard the true location entirely, causing the whole pipeline to fail regardless of how good Level-1 and Level-2 are.

## 13. Preserving the Existing NCC Implementation

- The existing `inference.py` (which contains the highly successful v2.5 NCC implementation) will be renamed to `inference_ncc.py` (or kept as `inference.py` while creating `inference_siamese.py`).
- We will create `compare_ncc_vs_siamese.py` to evaluate both algorithms on the exact same `evaluate.py` test sets, ensuring a fair, reproducible baseline comparison for the hackathon presentation.
