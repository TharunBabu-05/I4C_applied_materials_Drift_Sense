#!/usr/bin/env python3
"""
Phase 2 Multi-Task Training Script — Drift-Sense
=================================================
Fine-tunes PyramidSiameseNetworkV2 from the Phase 1 checkpoint
(best_model_level1.pth) on Phase 2 data with 3 simultaneous objectives:

  L_total = L_triplet  +  λ_scale * L_scale  +  λ_rot * L_rotation

Data format expected (from phase2_generator_60/generate_phase2_60_dataset.py):
  Each pair_XXXX folder contains:
    reference.png       -- clean reference crop
    search.png          -- full search image (may be degraded / absent / RGB)
    groundtruth.json    -- keys: target_x, target_y, theta, scale, found, set

Usage:
  python train_phase2.py \
      --data_dir ../../phase2_data \
      --resume   ../../final_model/best_model_level1.pth \
      --checkpoint_dir ./checkpoints_phase2 \
      --epochs 30 --lr 5e-4 --freeze_encoder_epochs 5
"""

import os, sys, json, random, time, argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

# ── Model import ──────────────────────────────────────────────────────────────
final_model_path = str((Path(__file__).parent.parent / "final_model").resolve())
if final_model_path not in sys.path:
    sys.path.insert(0, final_model_path)
from models.pyramid_siamese import PyramidSiameseNetwork


# =============================================================================
# Dataset
# =============================================================================
class Phase2Dataset(Dataset):
    """
    Loads (reference, positive_crop, scale_gt, theta_gt, found_gt)
    from the Phase 2 generator output.
    Only pairs where found=1 are used for pose regression training.
    All pairs (found=0 too) are used for contrastive/triplet training.
    """

    def __init__(self, data_dir: str, split_sets=("Set A", "Set B"),
                 include_absent: bool = True, img_size: int = 100):
        self.img_size = img_size
        self.samples = []

        data_dir = Path(data_dir)
        pair_dirs = sorted(data_dir.glob("pair_*"))

        for pair_dir in pair_dirs:
            gt_path = pair_dir / "groundtruth.json"
            ref_path = pair_dir / "reference.png"
            srch_path = pair_dir / "search.png"
            if not (gt_path.exists() and ref_path.exists() and srch_path.exists()):
                continue

            with open(gt_path) as f:
                gt = json.load(f)

            set_name = gt.get("set", "Set A")
            found = gt.get("found", 1)
            if not include_absent and not found:
                continue

            self.samples.append({
                "ref":   str(ref_path),
                "srch":  str(srch_path),
                "gt_x":  float(gt.get("target_x", 500)),
                "gt_y":  float(gt.get("target_y", 500)),
                "scale": float(gt.get("scale", 10.0)),
                "theta": float(gt.get("theta", 0.0)),
                "found": int(found),
                "set":   set_name,
            })

        print(f"[Phase2Dataset] Loaded {len(self.samples)} pairs from {data_dir}")

    def _load_gray(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ref = self._load_gray(s["ref"])  # 100x100
        
        srch_full = Image.open(s["srch"]).convert("L")
        srch_arr = np.array(srch_full, dtype=np.float32) / 255.0
        h, w = srch_arr.shape
        
        if s["found"] == 1:
            cx, cy = int(s["gt_x"]), int(s["gt_y"])
            scale = float(s["scale"])
            theta = float(s["theta"])
        else:
            # Random crop for absent pairs
            cx = random.randint(50, w - 50)
            cy = random.randint(50, h - 50)
            scale = 10.0
            theta = 0.0
            
        patch_size = int(round(1000.0 / scale))
        half_p = patch_size // 2
        
        y0, y1 = cy - half_p, cy - half_p + patch_size
        x0, x1 = cx - half_p, cx - half_p + patch_size
        
        # padding if outside
        pad_t, pad_b = max(0, -y0), max(0, y1 - h)
        pad_l, pad_r = max(0, -x0), max(0, x1 - w)
        y0_c, y1_c = max(0, y0), min(h, y1)
        x0_c, x1_c = max(0, x0), min(w, x1)
        
        crop = srch_arr[y0_c:y1_c, x0_c:x1_c]
        if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
            crop = np.pad(crop, ((pad_t, pad_b), (pad_l, pad_r)), mode='reflect')
            
        crop_100 = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        if abs(theta) > 1e-3:
            M = cv2.getRotationMatrix2D((self.img_size/2, self.img_size/2), -theta, 1.0)
            crop_100 = cv2.warpAffine(crop_100, M, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        ref_t  = torch.tensor(ref).unsqueeze(0)       # [1, 100, 100]
        srch_t = torch.tensor(crop_100).unsqueeze(0)  # [1, 100, 100]

        return {
            "ref":   ref_t,
            "srch":  srch_t,
            "scale": torch.tensor([s["scale"]], dtype=torch.float32),
            "theta": torch.tensor([s["theta"]], dtype=torch.float32),
            "found": torch.tensor([s["found"]], dtype=torch.float32),
        }


# =============================================================================
# SEM Augmentation (same as Phase 1 training — kept for noise invariance)
# =============================================================================
class SEMAugment:
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        # Poisson
        if random.random() < 0.6:
            scale = random.uniform(0.5, 1.5)
            noisy = torch.poisson(torch.clamp(x * 255.0 * scale, min=0)) / (255.0 * scale)
            x = torch.clamp(noisy, 0, 1)
        # Gaussian
        if random.random() < 0.6:
            x = torch.clamp(x + torch.randn_like(x) * random.uniform(0.01, 0.05), 0, 1)
        # Brightness jitter
        if random.random() < 0.5:
            gain = 1.0 + random.uniform(-0.15, 0.15)
            x = torch.clamp(x * gain, 0, 1)
        return x


# =============================================================================
# Training loop
# =============================================================================
def run_epoch(model, loader, optimizer, device, augment, train: bool,
              lambda_scale=0.5, lambda_rot=0.5, freeze_encoder: bool = False):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.5, p=2)
    mse = nn.MSELoss()

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            ref  = batch["ref"].to(device)
            srch = batch["srch"].to(device)
            scale_gt = batch["scale"].to(device)   # [B, 1]
            theta_gt = batch["theta"].to(device)   # [B, 1]
            found    = batch["found"].to(device)   # [B, 1]

            if train and augment:
                ref  = augment(ref)
                srch = augment(srch)

            if train:
                optimizer.zero_grad(set_to_none=True)

            ref_emb  = model.encoder(ref)
            srch_emb = model.encoder(srch)

            # ── Contrastive loss (positive pairs only for now) ──────────────
            # Simple contrastive: minimise distance for found=1, maximise for found=0
            sim = model.compute_similarity(ref_emb, srch_emb)          # [B]
            found_flat = found.squeeze(1)                                # [B]
            # Positive loss: found pairs should be similar (sim → 1)
            pos_mask = (found_flat == 1.0)
            ref_emb   = model.encoder(ref)
            srch_emb  = model.encoder(srch)
            sim_scores = model.compute_similarity(ref_emb, srch_emb)
            
            # 1. Contrastive Loss (Margin=1.0)
            # found=1 -> pull together (dist=0), found=0 -> push apart (dist>margin)
            dists = 1.0 - sim_scores
            loss = torch.where(
                found.squeeze(1) > 0.5,
                dists,
                torch.clamp(1.0 - dists, min=0.0)
            ).mean()

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item() * ref.size(0)
            n += ref.size(0)

    return total_loss / max(n, 1)


# =============================================================================
# Main
# =============================================================================
def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = Phase2Dataset(args.data_dir, include_absent=True)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = PyramidSiameseNetwork(embedding_dim=128, encoder_type="resnet").to(device)
    if args.resume and os.path.exists(args.resume):
        print(f"Loaded Phase 1 weights from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Missing keys (new heads):   {missing}")
        print(f"  Unexpected keys (ignored):  {unexpected}")
    else:
        print("[WARNING] No checkpoint found — training from scratch!")

    # We fine-tune the whole model
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    def make_optimizer(freeze_encoder: bool):
        if freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
            return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
            return optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    augment = SEMAugment(p=0.7)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        freeze = (epoch <= args.freeze_encoder_epochs)
        if epoch == 1 or epoch == args.freeze_encoder_epochs + 1:
            optimizer = make_optimizer(freeze_encoder=freeze)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.freeze_encoder_epochs)
            )

        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, augment,
                                train=True, freeze_encoder=freeze)
        val_loss   = run_epoch(model, val_loader,   optimizer, device, None,
                                train=False, freeze_encoder=False)
        if not freeze:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | lr={lr:.2e} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(args.checkpoint_dir, "best_model_phase2.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  --> New best val_loss={val_loss:.4f}. Saved: {ckpt}")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Multi-Task Training")
    parser.add_argument("--data_dir",             type=str, required=True,
                        help="Path to generated Phase 2 dataset directory")
    parser.add_argument("--resume",               type=str,
                        default="../../final_model/best_model_level1.pth",
                        help="Phase 1 checkpoint to fine-tune from")
    parser.add_argument("--checkpoint_dir",       type=str, default="./checkpoints_phase2")
    parser.add_argument("--encoder",              type=str, default="resnet",
                        choices=["resnet", "mobilenet"])
    parser.add_argument("--epochs",               type=int, default=30)
    parser.add_argument("--batch_size",           type=int, default=32)
    parser.add_argument("--lr",                   type=float, default=5e-4)
    parser.add_argument("--freeze_encoder_epochs",type=int, default=5,
                        help="Train only heads for this many epochs, then unfreeze encoder")
    parser.add_argument("--seed",                 type=int, default=42)
    main(parser.parse_args())
