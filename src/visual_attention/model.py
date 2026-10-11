"""Behavior Cloning policy network, loss, checkpointing and a numpy-facing policy wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Normalizer


class GazePolicy(nn.Module):
    """Two-headed policy: saliency-crop CNN + feature MLP -> shared trunk -> movement and dwell heads.

    With ``use_saliency=False`` (the ablation / Baseline B) the CNN branch is not built at all,
    so the model cannot see the saliency crop. The shuffled-crop control uses this same full
    architecture; only its *data* differ (see ``VARIANTS``).
    """

    def __init__(self, n_features: int, conv_channels=(16, 32, 32), feature_hidden: int = 64,
                 trunk_hidden: int = 128, dropout: float = 0.1, use_saliency: bool = True,
                 cnn_pool: int | None = None, crop_size: int = 84):
        super().__init__()
        self.use_saliency = use_saliency
        self.kwargs = dict(n_features=n_features, conv_channels=list(conv_channels),
                           feature_hidden=feature_hidden, trunk_hidden=trunk_hidden,
                           dropout=dropout, use_saliency=use_saliency, cnn_pool=cnn_pool,
                           crop_size=crop_size)
        if use_saliency:
            c1, c2, c3 = conv_channels
            side = crop_size
            for _ in range(3):  # spatial size after each stride-2 conv
                side = (side + 1) // 2
            pool = [nn.AdaptiveAvgPool2d(cnn_pool)] if cnn_pool else []
            cells = (cnn_pool or side) ** 2
            self.cnn = nn.Sequential(
                nn.Conv2d(1, c1, 5, stride=2, padding=2), nn.ReLU(),   # 84 -> 42
                nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),  # 42 -> 21
                nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),  # 21 -> 11
                *pool, nn.Flatten(),  # no pooling by default: keeps where things are in the crop
                nn.Linear(c3 * cells, trunk_hidden), nn.ReLU(),
            )
        self.feat = nn.Sequential(nn.Linear(n_features, feature_hidden), nn.ReLU())
        in_dim = feature_hidden + (trunk_hidden if use_saliency else 0)
        self.trunk = nn.Sequential(nn.Linear(in_dim, trunk_hidden), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU())
        self.move_head = nn.Linear(trunk_hidden, 2)
        self.dwell_head = nn.Linear(trunk_hidden, 1)

    def forward(self, features: torch.Tensor, saliency: torch.Tensor | None = None):
        """
        Args:
            features: ``(B, F)`` z-scored scalar features.
            saliency: ``(B, 1, S, S)`` saliency crop in [0, 1]; required iff ``use_saliency``.
        Returns:
            ``movement`` ``(B, 2)`` z-scored (dx, dy) and ``dwell`` ``(B, 1)`` z-scored log-dwell.
        """
        h = self.feat(features)
        if self.use_saliency:
            if saliency is None:
                raise ValueError("this model needs the saliency input")
            h = torch.cat([self.cnn(saliency), h], dim=1)
        z = self.trunk(h)
        return self.move_head(z), self.dwell_head(z)


# variant -> (build the CNN branch / feed a saliency crop, crops shuffled across transitions)
# shuffled_saliency is a control: identical architecture and training to "full", but each
# transition is paired with the saliency crop of a different, randomly chosen transition of the
# same split. The input has the same statistics as real crops but carries no information about
# the current gaze position.
VARIANTS = {"full": (True, False), "no_saliency": (False, False), "shuffled_saliency": (True, True)}


def build_model(cfg: dict, n_features: int, variant: str) -> GazePolicy:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    use_saliency, _ = VARIANTS[variant]
    m = cfg["model"]
    return GazePolicy(n_features, m["conv_channels"], m["feature_hidden"], m["trunk_hidden"],
                      m["dropout"], use_saliency, m.get("cnn_pool"), cfg["observation"]["crop_size"])


def bc_loss(move_pred, dwell_pred, move_target, dwell_target, lambda_movement=1.0, lambda_dwell=1.0):
    """``total = lambda_m * MSE(movement_z) + lambda_d * MSE(log-dwell_z)``; returns (total, move, dwell)."""
    lm = F.mse_loss(move_pred, move_target)
    ld = F.mse_loss(dwell_pred, dwell_target)
    return lambda_movement * lm + lambda_dwell * ld, lm, ld


def save_checkpoint(path, model: GazePolicy, normalizer: Normalizer, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_kwargs": model.kwargs,
                "normalizer": normalizer.to_dict(), **(extra or {})}, path)


def load_checkpoint(path, device="cpu") -> tuple[GazePolicy, Normalizer, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GazePolicy(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Normalizer.from_dict(ckpt["normalizer"]), ckpt


class Policy:
    """Maps a raw environment observation to (displacement, dwell_ms) using a trained model."""

    def __init__(self, model: GazePolicy, normalizer: Normalizer, device="cpu"):
        self.model, self.norm, self.device = model.eval(), normalizer, device

    @torch.no_grad()
    def act(self, obs: dict) -> tuple[np.ndarray, float]:
        f = torch.from_numpy(self.norm.features(obs["features"])[None]).to(self.device)
        s = None
        if self.model.use_saliency:
            s = torch.from_numpy(obs["saliency"][None].astype(np.float32)).to(self.device)
        mz, dz = self.model(f, s)
        move = self.norm.inv_movement(mz.cpu().numpy()[0])
        dwell = float(self.norm.inv_dwell(dz.cpu().numpy()[0, 0]))
        return move, dwell
