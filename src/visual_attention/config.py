"""Configuration loading, overrides, seeding and device selection."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply ``key.subkey=value`` overrides (values parsed as YAML)."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        set_by_path(cfg, *parse_override(item))
    return cfg


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must look like key.subkey=value, got {item!r}")
    key, value = item.split("=", 1)
    return key.strip(), yaml.safe_load(value)


def set_by_path(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Unknown config section {p!r} in override {key!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key {key!r}")
    node[parts[-1]] = value


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(cfg: dict, name: str) -> Path:
    """Absolute path for ``cfg['paths'][name]`` (relative paths are taken from the project root)."""
    p = Path(cfg["paths"][name])
    return p if p.is_absolute() else PROJECT_ROOT / p


def results_subdir(cfg: dict, sub: str) -> Path:
    d = resolve_path(cfg, "results_dir") / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto"):
    """Return a torch.device. ``auto`` picks CUDA, then MPS, then CPU."""
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available")
    return torch.device(name)

