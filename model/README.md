# Drift-Sense Siamese Multi-Scale Network

This directory contains the PyTorch implementation of the Drift-Sense Siamese Multi-Scale Network for robust SEM image localization.

## 1. Architecture

The network builds upon the highly successful 3-Level Multi-Scale NCC Pyramid (v2.5) by replacing the classical Cross-Correlation operator with a trainable Siamese Neural Network. 

- **Encoder:** A lightweight custom ResNet (4 stages, 128-D output) designed for extremely fast inference on CPU/Edge devices.
- **Level 0 (Coarse):** Extracts overlapping 50x50 patches from a 2x downscaled search image. Extracts Top-K candidates.
- **Level 1 (Nominal):** Refines candidates using 100x100 patches from the original search image. Fuses Level-0 and Level-1 similarities.
- **Level 2 (Fine Refinement):** Uses a sub-pixel regression head for ultra-precise `Δx, Δy` offsets relative to the best Level-1 candidate.

## 2. Dataset Generation

We use the advanced `standalone_dataset_generator` provided by your colleague. It includes 60 CAD semiconductor layout subscripts, 17 degradation engines, and explicitly generates periodic hard-negative shifts to prevent the network from memorizing patterns.

**Run Generation:**
```bash
python standalone_dataset_generator/generate_dataset.py --num_pairs 2500 --output_dir ./data
```

## 3. Training

The Siamese network is trained using **Margin-based Contrastive Loss**.
It learns to embed the reference patch and a perfectly aligned candidate patch close to each other, while pushing apart the reference and a slightly misaligned periodic neighbor (Hard Negative).

**Run Training:**
```bash
python training/train_siamese.py --data_dir ./data --checkpoint_dir ./checkpoints --epochs 30 --batch_size 32
```

### Hyperparameters
- **Optimizer:** AdamW
- **Learning Rate:** 1e-3 (Cosine Annealing)
- **Margin:** 1.0
- **Embedding Dim:** 128
- **Batch Size:** 32

## 4. Inference

The inference pipeline mirrors the classical NCC pipeline but uses the trained neural encoder for similarity scoring. It runs quickly by batching all patch extractions.

**Run Inference:**
```bash
python inference/inference_siamese.py --reference path/to/ref.png --search path/to/search.png --checkpoint checkpoints/best_model_level1.pth
```

## 5. Evaluation & Comparison

We evaluate against the existing highly successful NCC baseline.

**Run Evaluation:**
```bash
python evaluation/compare_ncc_vs_siamese.py --data_dir ./data/test --checkpoint checkpoints/best_model_level1.pth
```

## 6. Visualization

Generate side-by-side visual reports of the prediction versus ground truth.

**Run Visualization:**
```bash
python inference/visualization.py --reference path/to/ref.png --search path/to/search.png --pred_x 500 --pred_y 500 --gt_x 502 --gt_y 504
```

## 7. Results & Limitations

**Expected Advantages:**
- Vastly improved robustness against high-amplitude shot noise.
- Tolerance against random structural defects (missing contacts) which would otherwise wreck normalized correlation scores.

**Limitations:**
- Training data is synthetic; out-of-distribution (OOD) shift is the largest risk when deploying on Applied Materials' real SEM images.
- A badly trained Siamese encoder might collapse (all patches map to similar embeddings), destroying the localization entirely.

## 8. Future FPGA / Edge Deployment

The encoder is heavily constrained (only 4 small residual blocks) yielding very low FLOPs and parameter counts.
- It can be easily exported via ONNX (`torch.onnx.export`).
- Suitable for INT8 quantization using TensorRT or OpenVINO.
- Since it doesn't use complex attention mechanisms or dense decoders, it can be seamlessly ported to FPGA accelerators for ultra-low latency inference on the actual wafer inspection tool.
