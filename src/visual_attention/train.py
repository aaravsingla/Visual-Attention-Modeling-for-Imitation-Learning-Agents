"""Behavior Cloning training with validation-based checkpoint selection and early stopping.

Usage (from the project root, after ``scripts/generate_data.py``)::

    python -m visual_attention.train --variant full
    python -m visual_attention.train --variant no_saliency --epochs 10 --lr 3e-4 --device cpu

(``python -m`` needs ``src`` on the path: ``pip install -e .`` or ``set PYTHONPATH=src``.)
The test split is never loaded here.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_device, resolve_path, results_subdir, save_config, set_seed
from .dataset import (Normalizer, TransitionDataset, load_obs_maps, load_splits, load_transitions,
                      split_indices)
from .model import VARIANTS, bc_loss, build_model, save_checkpoint


def checkpoint_path(cfg: dict, variant: str, seed: int | None = None) -> Path:
    seed = cfg["seed"] if seed is None else seed
    return resolve_path(cfg, "checkpoint_dir") / f"{variant}_seed{seed}" / "best.pt"


def make_datasets(cfg: dict, use_saliency: bool, data: dict | None = None, splits: dict | None = None,
                  obs_maps: dict | None = None, which=("train", "val"), normalizer: Normalizer | None = None,
                  shuffle_crops: bool = False):
    """Datasets for the requested splits, normalised with *training-set* statistics.

    ``normalizer``: pass a checkpoint's stored normaliser at evaluation time; otherwise it is fitted
    on the training split. ``use_saliency``: whether items carry a saliency crop.
    ``shuffle_crops`` (control): pair every transition with the crop of another transition of the
    same split (a fixed random permutation per split).
    """
    data = load_transitions(cfg) if data is None else data
    splits = load_splits(cfg) if splits is None else splits
    norm = normalizer if normalizer is not None else Normalizer.fit(data, split_indices(data, splits["train"]))
    if use_saliency and obs_maps is None:
        obs_maps = load_obs_maps(cfg, [d for s in which for d in splits[s]])
    out = {}
    for s in which:
        idx = split_indices(data, splits[s])
        perm = None
        if shuffle_crops:
            perm = np.random.default_rng([cfg["seed"], 31, ["train", "val", "test"].index(s)]).permutation(len(idx))
        out[s] = TransitionDataset(data, idx, obs_maps, norm, cfg["observation"], use_saliency, perm)
    return out, norm


def _to(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, device, lambdas, optimizer=None) -> dict:
    """One pass over ``loader``. Trains if ``optimizer`` is given, otherwise evaluates without grads."""
    training = optimizer is not None
    model.train(training)
    tot = {"loss": 0.0, "movement_loss": 0.0, "dwell_loss": 0.0}
    n = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            b = _to(batch, device)
            move, dwell = model(b["features"], b.get("saliency"))
            loss, lm, ld = bc_loss(move, dwell, b["movement"], b["dwell"], *lambdas)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss {loss.item()}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            k = len(b["features"])
            n += k
            tot["loss"] += loss.item() * k
            tot["movement_loss"] += lm.item() * k
            tot["dwell_loss"] += ld.item() * k
    return {key: v / max(n, 1) for key, v in tot.items()}


def train(cfg: dict, variant: str = "full", log=print, data=None, splits=None, obs_maps=None) -> dict:
    """Train one variant; saves the best-validation checkpoint and a per-epoch CSV log."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {list(VARIANTS)}")
    tcfg = cfg["training"]
    seed = cfg["seed"]
    set_seed(seed)
    if tcfg.get("num_threads"):
        torch.set_num_threads(int(tcfg["num_threads"]))
    device = resolve_device(tcfg["device"])
    use_sal, shuffle = VARIANTS[variant]

    ds, norm = make_datasets(cfg, use_sal, data, splits, obs_maps, shuffle_crops=shuffle)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(ds["train"], batch_size=tcfg["batch_size"], shuffle=True, generator=gen)
    val_loader = DataLoader(ds["val"], batch_size=1024, shuffle=False)

    model = build_model(cfg, ds["train"].features.shape[1], variant).to(device)
    assert model.use_saliency == use_sal
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    lambdas = (tcfg["lambda_movement"], tcfg["lambda_dwell"])

    ckpt = checkpoint_path(cfg, variant)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, ckpt.parent / "config.yaml")
    log_path = results_subdir(cfg, "logs") / f"train_{variant}_seed{seed}.csv"

    history, best, best_epoch, bad = [], float("inf"), -1, 0
    t0 = time.perf_counter()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[{variant}] device={device} params={n_params} train={len(ds['train'])} val={len(ds['val'])}")
    for epoch in range(1, tcfg["max_epochs"] + 1):
        tr = run_epoch(model, train_loader, device, lambdas, opt)
        va = run_epoch(model, val_loader, device, lambdas)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
               **{f"val_{k}": v for k, v in va.items()}, "seconds": time.perf_counter() - t0}
        history.append(row)
        improved = va["loss"] < best - 1e-5
        if improved:
            best, best_epoch, bad = va["loss"], epoch, 0
            save_checkpoint(ckpt, model, norm, {"variant": variant, "epoch": epoch, "val_loss": best,
                                                "seed": seed})
        else:
            bad += 1
        log(f"[{variant}] epoch {epoch:2d} train {tr['loss']:.4f} (move {tr['movement_loss']:.4f} "
            f"dwell {tr['dwell_loss']:.4f}) | val {va['loss']:.4f} (move {va['movement_loss']:.4f} "
            f"dwell {va['dwell_loss']:.4f}){' *' if improved else ''}")
        if bad >= tcfg["patience"]:
            log(f"[{variant}] early stopping after epoch {epoch} (best epoch {best_epoch})")
            break

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    summary = {"variant": variant, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best,
               "epochs_run": len(history), "train_seconds": round(time.perf_counter() - t0, 1),
               "n_params": n_params, "checkpoint": str(ckpt), "log": str(log_path), "device": str(device)}
    with open(ckpt.parent / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["history"] = history
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the Behavior Cloning gaze policy")
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--variant", default="full", choices=list(VARIANTS))
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int, help="maximum number of epochs")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device", help="auto | cpu | cuda | mps")
    ap.add_argument("--checkpoint-dir", help="where checkpoints are written")
    ap.add_argument("--results-dir", help="where logs / metrics / figures are written")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="other config overrides")
    return ap


def apply_cli(cfg: dict, args) -> dict:
    t = cfg["training"]
    for attr, key in (("batch_size", "batch_size"), ("lr", "lr"), ("epochs", "max_epochs"), ("device", "device")):
        if getattr(args, attr, None) is not None:
            t[key] = getattr(args, attr)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "checkpoint_dir", None):
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir
    if getattr(args, "results_dir", None):
        cfg["paths"]["results_dir"] = args.results_dir
    return cfg


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    cfg = apply_cli(load_config(args.config, args.set), args)
    s = train(cfg, args.variant)
    s.pop("history")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
