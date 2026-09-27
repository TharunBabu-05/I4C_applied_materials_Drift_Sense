#!/usr/bin/env python3
"""
PyramidSiameseNetworkV2 — Phase 2 Extended Model
=================================================
Extends the Phase 1 PyramidSiameseNetwork with two lightweight regression heads:
  - scale_head  : predicts the down-scaling factor  ∈ [8, 12]
  - rotation_head: predicts the rotation angle (theta) ∈ [-5°, +5°]

The shared encoder weights are kept compatible with best_model_level1.pth
(Phase 1 checkpoint) so we can fine-tune rather than train from scratch.

Usage (fine-tune from Phase 1 checkpoint):
    model = PyramidSiameseNetworkV2(encoder_type='resnet')
    # Load Phase 1 weights (strict=False to allow new head params)
    state = torch.load('best_model_level1.pth', map_location='cpu')
    model.load_state_dict(state, strict=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siamese_encoder import SiameseEncoder, MobileNetV3SiameseEncoder


class PyramidSiameseNetworkV2(nn.Module):
    """
    Phase 2 extension of PyramidSiameseNetwork.
    Adds:
      - scale_head     : concat(ref_emb, cand_emb) → scalar scale ∈ [8, 12]
      - rotation_head  : concat(ref_emb, cand_emb) → scalar theta ∈ [-5, 5] deg
    """

    def __init__(self, embedding_dim: int = 128, encoder_type: str = 'resnet'):
        super().__init__()

        # ── Shared encoder (same as Phase 1, weights are transferable) ──────
        if encoder_type == 'mobilenet':
            self.encoder = MobileNetV3SiameseEncoder(embedding_dim=embedding_dim)
        else:
            self.encoder = SiameseEncoder(embedding_dim=embedding_dim)

        # ── Phase 1 inherited heads (unchanged weights) ──────────────────────
        self.fusion_weight_l0 = nn.Parameter(torch.tensor([0.35]))
        self.fusion_weight_l1 = nn.Parameter(torch.tensor([0.65]))
        self.refinement_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

        # ── NEW: Phase 2 pose regression heads ───────────────────────────────
        # Input: concatenation of ref_emb and candidate_emb → 2 * embedding_dim
        pose_in = embedding_dim * 2

        self.scale_head = nn.Sequential(
            nn.Linear(pose_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),   # output: normalized scale
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(pose_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),   # output: normalized theta
        )

    # ── Forward (for training) ────────────────────────────────────────────────
    def forward(self, ref: torch.Tensor, candidate: torch.Tensor):
        """
        Returns (ref_emb, cand_emb) — training loop uses these directly.
        """
        return self.encoder(ref), self.encoder(candidate)

    # ── Similarity (same as Phase 1) ──────────────────────────────────────────
    def compute_similarity(self, ref_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """Cosine similarity (embeddings are already L2-normalised by encoder)."""
        return torch.sum(ref_emb * cand_emb, dim=1)

    # ── Pose estimation (NEW) ─────────────────────────────────────────────────
    def predict_pose(self, ref_emb: torch.Tensor, cand_emb: torch.Tensor):
        """
        Predict scale and rotation from a reference / candidate embedding pair.

        Returns:
            scale_pred  (Tensor, shape [B]): predicted scale, clamped to [8, 12]
            theta_pred  (Tensor, shape [B]): predicted theta in degrees, clamped to [-5, 5]
        """
        concat = torch.cat([ref_emb, cand_emb], dim=1)   # [B, 2*D]

        # Scale: raw output → sigmoid → remap to [8, 12]
        scale_raw = self.scale_head(concat).squeeze(1)     # [B]
        scale_pred = 8.0 + 4.0 * torch.sigmoid(scale_raw) # [8, 12]

        # Rotation: raw output → tanh → remap to [-5, 5]
        theta_raw = self.rotation_head(concat).squeeze(1)  # [B]
        theta_pred = 5.0 * torch.tanh(theta_raw)           # [-5, 5]

        return scale_pred, theta_pred

    # ── Fine-level refinement (same as Phase 1) ───────────────────────────────
    def refine_fine_candidate(self, ref_emb: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([ref_emb, cand_emb], dim=1)
        return self.refinement_head(concat)
