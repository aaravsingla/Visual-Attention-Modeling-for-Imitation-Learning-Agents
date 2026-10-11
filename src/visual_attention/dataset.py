"""Expert demonstration generation, storage, document-level splits and the PyTorch Dataset.

Storage (all under ``<data_dir>``):

* ``raw/doc_XXXX.png`` + ``raw/doc_XXXX.json``  document image and layout.
* ``processed/transitions.npz``  one row per expert transition (fixation t -> t+1):

  ===============  =========  ===============================================================
  key              shape      meaning
  ===============  =========  ===============================================================
  doc_id           (N,)       document id
  traj_id          (N,)       trajectory index within the document
  t                (N,)       time step (fixation index)
  gaze             (N, 2)     current fixation, normalised [0, 1]
  next_gaze        (N, 2)     expert's next fixation (target only, never an input)
  displacement     (N, 2)     next_gaze - gaze (movement target)
  dwell            (N,)       expert dwell of the current fixation, synthetic ms (target)
  dwell_clean      (N,)       dwell before the noise term (analysis only, never an input)
  features         (N, F)     scalar observation features at ``gaze`` (environment.FEATURE_NAMES)
  event            (N,)       saccade type (expert.EVENT_NAMES), for error analysis only
  word             (N,)       fixated word index, for analysis only
  ===============  =========  ===============================================================

  The saliency-crop part of the observation is a deterministic function of the document
  image and ``gaze`` and is recomputed on the fly (``environment.extract_crop``).
  The last fixation of each trajectory has no next fixation and is therefore not a row.
* ``processed/splits.json``  document ids of the train / val / test splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import resolve_path
from .document_generator import generate_document, load_document
from .environment import FEATURE_NAMES, extract_crop, local_features, observation_map
from .expert import ReadingExpert
from .saliency import compute_maps

ARRAY_KEYS = ("doc_id", "traj_id", "t", "gaze", "next_gaze", "displacement", "dwell",
              "dwell_clean", "features", "event", "word")


def data_paths(cfg: dict) -> tuple[Path, Path]:
    root = resolve_path(cfg, "data_dir")
    return root / "raw", root / "processed"


def trajectory_rng(seed: int, doc_id: int, traj_id: int) -> np.random.Generator:
    return np.random.default_rng([seed, doc_id, traj_id, 7])


def build_transitions(doc, maps, cfg: dict, seed: int) -> dict:
    """All expert transitions for one document as a dict of arrays."""
    expert = ReadingExpert(doc, maps, cfg["expert"])
    rows = {k: [] for k in ARRAY_KEYS}
    for k in range(cfg["expert"]["trajectories_per_document"]):
        tr = expert.sample_trajectory(trajectory_rng(seed, doc.doc_id, k))
        # cast once so stored gaze, stored features and on-the-fly crops use identical coordinates
        fix = tr["fixations"].astype(np.float32)
        for t in range(len(fix) - 1):
            rows["doc_id"].append(doc.doc_id)
            rows["traj_id"].append(k)
            rows["t"].append(t)
            rows["gaze"].append(fix[t])
            rows["next_gaze"].append(fix[t + 1])
            rows["displacement"].append(fix[t + 1] - fix[t])
            rows["dwell"].append(tr["dwell"][t])
            rows["dwell_clean"].append(tr["dwell_clean"][t])
            rows["features"].append(local_features(maps, fix[t], cfg["observation"]))
            rows["event"].append(tr["event"][t])
            rows["word"].append(tr["word"][t])
    return rows


def split_documents(doc_ids, fractions: dict, seed: int) -> dict[str, list[int]]:
    """Random document-level split. Every document lands in exactly one split."""
    ids = np.array(sorted(set(int(d) for d in doc_ids)))
    perm = np.random.default_rng([seed, 99]).permutation(ids)
    n = len(ids)
    n_test = max(1, int(round(fractions["test"] * n)))
    n_val = max(1, int(round(fractions["val"] * n)))
    if n - n_test - n_val < 1:
        raise ValueError(f"too few documents ({n}) for a train/val/test split")
    splits = {"test": sorted(perm[:n_test].tolist()),
              "val": sorted(perm[n_test:n_test + n_val].tolist()),
              "train": sorted(perm[n_test + n_val:].tolist())}
    assert_disjoint(splits)
    return splits


def assert_disjoint(splits: dict[str, list[int]]) -> None:
    names = list(splits)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            overlap = set(splits[names[a]]) & set(splits[names[b]])
            if overlap:
                raise ValueError(f"documents {sorted(overlap)} appear in both {names[a]} and {names[b]}")


def generate_dataset(cfg: dict) -> dict:
    """Generate documents and expert demonstrations, write them to ``data_dir``. Returns a summary.

    Uses ``cfg["data_seed"]`` (documents, expert trajectories, split), which is separate from the
    training seed ``cfg["seed"]``.
    """
    seed = cfg.get("data_seed", cfg["seed"])
    raw_dir, proc_dir = data_paths(cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in ARRAY_KEYS}
    for doc_id in range(cfg["document"]["n_documents"]):
        doc = generate_document(doc_id, cfg["document"], seed)
        doc.save(raw_dir)
        maps = compute_maps(doc.image, cfg["saliency"])
        r = build_transitions(doc, maps, cfg, seed)
        for k in ARRAY_KEYS:
            rows[k].extend(r[k])
    data = {
        "doc_id": np.array(rows["doc_id"], dtype=np.int64),
        "traj_id": np.array(rows["traj_id"], dtype=np.int64),
        "t": np.array(rows["t"], dtype=np.int64),
        "gaze": np.array(rows["gaze"], dtype=np.float32),
        "next_gaze": np.array(rows["next_gaze"], dtype=np.float32),
        "displacement": np.array(rows["displacement"], dtype=np.float32),
        "dwell": np.array(rows["dwell"], dtype=np.float32),
        "dwell_clean": np.array(rows["dwell_clean"], dtype=np.float32),
        "features": np.array(rows["features"], dtype=np.float32),
        "event": np.array(rows["event"], dtype=np.int64),
        "word": np.array(rows["word"], dtype=np.int64),
    }
    np.savez_compressed(proc_dir / "transitions.npz", feature_names=np.array(FEATURE_NAMES), **data)
    splits = split_documents(data["doc_id"], cfg["split"], seed)
    with open(proc_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({"data_seed": seed, "fractions": cfg["split"], **splits}, f, indent=2)
    n_traj = len(set(zip(data["doc_id"].tolist(), data["traj_id"].tolist())))
    return {"n_documents": cfg["document"]["n_documents"], "n_trajectories": n_traj,
            "n_transitions": int(len(data["dwell"])),
            **{f"n_{s}_docs": len(splits[s]) for s in ("train", "val", "test")}}


def load_transitions(cfg: dict) -> dict:
    _, proc_dir = data_paths(cfg)
    with np.load(proc_dir / "transitions.npz") as z:
        return {k: z[k] for k in z.files}


def load_splits(cfg: dict) -> dict[str, list[int]]:
    _, proc_dir = data_paths(cfg)
    with open(proc_dir / "splits.json", "r", encoding="utf-8") as f:
        s = json.load(f)
    splits = {k: s[k] for k in ("train", "val", "test")}
    assert_disjoint(splits)
    return splits


def split_indices(data: dict, doc_ids) -> np.ndarray:
    return np.flatnonzero(np.isin(data["doc_id"], np.asarray(list(doc_ids))))


def load_obs_maps(cfg: dict, doc_ids) -> dict[int, np.ndarray]:
    """Padded crop source maps (environment.observation_map) recomputed from the saved images."""
    raw_dir, _ = data_paths(cfg)
    out = {}
    for d in sorted(set(int(i) for i in doc_ids)):
        doc = load_document(raw_dir, d)
        out[d] = observation_map(compute_maps(doc.image, cfg["saliency"]).saliency, cfg["observation"])
    return out


class Normalizer:
    """Train-set statistics. Movement targets are z-scored per axis; dwell is z-scored in log space.

    Inverse transforms: ``displacement = z * move_std + move_mean`` and
    ``dwell_ms = exp(z * logdwell_std + logdwell_mean)`` (always positive).
    """

    def __init__(self, feat_mean, feat_std, move_mean, move_std, logdwell_mean, logdwell_std):
        self.feat_mean = np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = np.asarray(feat_std, dtype=np.float32)
        self.move_mean = np.asarray(move_mean, dtype=np.float32)
        self.move_std = np.asarray(move_std, dtype=np.float32)
        self.logdwell_mean = float(logdwell_mean)
        self.logdwell_std = float(logdwell_std)

    @classmethod
    def fit(cls, data: dict, idx: np.ndarray) -> "Normalizer":
        f = data["features"][idx]
        m = data["displacement"][idx]
        ld = np.log(data["dwell"][idx].astype(np.float64))
        return cls(f.mean(0), np.maximum(f.std(0), 1e-6), m.mean(0), np.maximum(m.std(0), 1e-6),
                   ld.mean(), max(ld.std(), 1e-6))

    def features(self, f):
        return (np.asarray(f, dtype=np.float32) - self.feat_mean) / self.feat_std

    def movement(self, m):
        return (np.asarray(m, dtype=np.float32) - self.move_mean) / self.move_std

    def dwell(self, d):
        return ((np.log(np.asarray(d, dtype=np.float64)) - self.logdwell_mean) / self.logdwell_std).astype(np.float32)

    def inv_movement(self, z):
        return np.asarray(z, dtype=np.float64) * self.move_std + self.move_mean

    def inv_dwell(self, z):
        return np.exp(np.asarray(z, dtype=np.float64) * self.logdwell_std + self.logdwell_mean)

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(**d)


class TransitionDataset(Dataset):
    """Model inputs and normalised targets for a subset of transitions.

    Each item: ``features (F,)``, ``movement (2,)``, ``dwell (1,)`` and, if ``use_saliency``,
    ``saliency (1, S, S)``. Inputs depend only on the document and the current gaze.

    ``crop_perm`` (control experiment only): item ``i`` gets the saliency crop of item
    ``crop_perm[i]`` instead of its own.
    """

    def __init__(self, data: dict, idx: np.ndarray, obs_maps: dict[int, np.ndarray] | None,
                 normalizer: Normalizer, cfg_obs: dict, use_saliency: bool = True,
                 crop_perm: np.ndarray | None = None):
        self.idx = np.asarray(idx)
        src = self.idx if crop_perm is None else self.idx[np.asarray(crop_perm)]
        self.crop_doc_id = data["doc_id"][src]
        self.crop_gaze = data["gaze"][src]
        self.features = normalizer.features(data["features"][self.idx])
        self.movement = normalizer.movement(data["displacement"][self.idx])
        self.dwell = normalizer.dwell(data["dwell"][self.idx])[:, None]
        self.obs_maps = obs_maps
        self.cfg_obs = cfg_obs
        self.use_saliency = use_saliency
        if use_saliency and obs_maps is None:
            raise ValueError("obs_maps are required when use_saliency=True")

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        item = {"features": torch.from_numpy(self.features[i]),
                "movement": torch.from_numpy(self.movement[i]),
                "dwell": torch.from_numpy(self.dwell[i])}
        if self.use_saliency:
            crop = extract_crop(self.obs_maps[int(self.crop_doc_id[i])], self.crop_gaze[i], self.cfg_obs)
            item["saliency"] = torch.from_numpy(crop[None].astype(np.float32))
        return item
