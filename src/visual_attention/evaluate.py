"""Held-out evaluation: dwell and movement metrics, baselines, saliency ablation, rollouts, figures.

Two kinds of evaluation are reported and kept apart:

* **Teacher-forced (one step ahead):** for every held-out expert transition the model sees the
  expert's *current* fixation and predicts the dwell of that fixation and the displacement to
  the next one. Dwell and next-fixation metrics use this.
* **Free-running rollout:** the model starts at the expert's first fixation and then moves
  using its *own* predicted displacements (the expert is never consulted again).

Usage::

    python -m visual_attention.evaluate [--config ...] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from .config import load_config, resolve_device, results_subdir, save_config
from .dataset import data_paths, load_obs_maps, load_splits, load_transitions, split_indices
from .document_generator import load_document
from .environment import ReadingEnv
from .expert import EVENT_NAMES
from .model import VARIANTS, Policy, load_checkpoint
from .saliency import compute_maps
from .train import checkpoint_path, make_datasets

MODEL_LABELS = {"full": "BC full (saliency + features)", "no_saliency": "BC features only (ablation)",
                "shuffled_saliency": "BC shuffled-crop control (same CNN, crops from other transitions)",
                "constant": "constant baseline (train means)"}
SHORT_LABELS = {"constant": "constant", "no_saliency": "BC features only",
                "shuffled_saliency": "BC shuffled crops", "full": "BC full"}
MODEL_ORDER = ["constant", "no_saliency", "shuffled_saliency", "full"]


# ----------------------------------------------------------------------------- metrics
def pearson(x, y) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined (n < 2 or a constant input)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if len(x) < 2:
        return None
    xc, yc = x - x.mean(), y - y.mean()
    den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
    if den < 1e-12 * max(1.0, len(x)):
        return None
    return float(np.clip((xc * yc).sum() / den, -1.0, 1.0))


def dwell_metrics(expert_ms, pred_ms) -> dict:
    """Dwell metrics in synthetic milliseconds (MAE / RMSE in ms; r and R^2 unitless)."""
    e, p = np.asarray(expert_ms, dtype=np.float64), np.asarray(pred_ms, dtype=np.float64)
    r2 = None if len(e) < 2 or np.allclose(e, e[0]) else float(r2_score(e, p))
    return {"n": int(len(e)), "pearson_r": pearson(e, p),
            "mae_ms": float(mean_absolute_error(e, p)),
            "rmse_ms": float(np.sqrt(mean_squared_error(e, p))), "r2": r2,
            "mean_expert_ms": float(e.mean()), "mean_pred_ms": float(p.mean())}


def displacement_errors(true_disp, pred_disp, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Euclidean next-fixation error per transition in normalised units and in pixels."""
    d = np.asarray(pred_disp, dtype=np.float64) - np.asarray(true_disp, dtype=np.float64)
    norm = np.sqrt((d ** 2).sum(1))
    px = np.sqrt((d[:, 0] * width) ** 2 + (d[:, 1] * height) ** 2)
    return norm, px


def movement_metrics(true_disp, pred_disp, width: int, height: int, events=None, in_text=None) -> dict:
    norm, px = displacement_errors(true_disp, pred_disp, width, height)
    out = {"n": int(len(norm)), "mean_err_norm": float(norm.mean()),
           "median_err_norm": float(np.median(norm)), "mean_err_px": float(px.mean()),
           "median_err_px": float(np.median(px))}
    if in_text is not None:
        out["pred_in_text_fraction"] = float(np.mean(in_text))
    if events is not None:
        events = np.asarray(events)
        out["by_event_mean_err_px"] = {
            name: (float(px[events == k].mean()) if np.any(events == k) else None)
            for k, name in enumerate(EVENT_NAMES)}
        out["by_event_n"] = {name: int(np.sum(events == k)) for k, name in enumerate(EVENT_NAMES)}
    return out


def in_text_mask(points, text_mask: np.ndarray) -> np.ndarray:
    """Whether normalised points (N, 2) fall on the text-region mask."""
    h, w = text_mask.shape
    p = np.clip(np.asarray(points, dtype=np.float64), 0, 1)
    cols = np.clip((p[:, 0] * w).astype(int), 0, w - 1)
    rows = np.clip((p[:, 1] * h).astype(int), 0, h - 1)
    return text_mask[rows, cols] > 0.5


# ----------------------------------------------------------------------------- predictions
@torch.no_grad()
def predict_dataset(model, dataset, normalizer, device) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced predictions for every item: displacement (N, 2) and dwell in ms (N,)."""
    model.eval()
    moves, dwells = [], []
    for b in DataLoader(dataset, batch_size=1024, shuffle=False):
        s = b["saliency"].to(device) if "saliency" in b else None
        m, d = model(b["features"].to(device), s)
        moves.append(m.cpu().numpy())
        dwells.append(d.cpu().numpy()[:, 0])
    return normalizer.inv_movement(np.concatenate(moves)), normalizer.inv_dwell(np.concatenate(dwells))


def constant_baseline(data: dict, train_idx: np.ndarray) -> dict:
    """Baseline A: training-set mean dwell and mean displacement (uses training labels only)."""
    return {"dwell_ms": float(data["dwell"][train_idx].mean()),
            "displacement": data["displacement"][train_idx].mean(0).astype(np.float64)}


class ConstantPolicy:
    def __init__(self, displacement, dwell_ms):
        self.d, self.dwell = np.asarray(displacement, dtype=np.float64), float(dwell_ms)

    def act(self, obs):
        return self.d.copy(), self.dwell


def rollout(policy, env: ReadingEnv, start, n_steps: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Free-running rollout: the policy's own displacements drive the gaze."""
    env.max_steps = n_steps
    obs = env.reset(start)
    done = False
    while not done:
        move, dwell = policy.act(obs)
        obs, done, _ = env.step(move, dwell)
    fix, dw = env.trajectory()
    return fix, dw, env.n_clipped


def scanpath_metrics(fix, doc, text_mask) -> dict:
    """Layout-based summary of one scanpath (layout is used for evaluation only)."""
    px = fix * np.array([doc.width, doc.height])
    covered = 0
    for line in doc.lines:
        x0, y0, x1, y1 = line.box
        inside = (px[:, 0] >= x0 - 4) & (px[:, 0] <= x1 + 4) & (px[:, 1] >= y0 - 4) & (px[:, 1] <= y1 + 4)
        covered += int(inside.any())
    words_hit = 0
    for word in doc.words:
        x0, y0, x1, y1 = word.box
        words_hit += int(np.any((px[:, 0] >= x0 - 3) & (px[:, 0] <= x1 + 3) & (px[:, 1] >= y0 - 3) & (px[:, 1] <= y1 + 3)))
    return {"in_text_fraction": float(in_text_mask(fix, text_mask).mean()),
            "word_coverage": words_hit / max(len(doc.words), 1),
            "line_coverage": covered / max(len(doc.lines), 1),
            "unique_positions": int(len(np.unique(np.round(fix, 3), axis=0)))}


# ----------------------------------------------------------------------------- main routine
def evaluate(cfg: dict, variants=("full", "no_saliency", "shuffled_saliency"), splits_to_eval=("val", "test"),
             make_figures: bool = True, log=print) -> dict:
    seed = cfg["seed"]
    device = resolve_device(cfg["training"]["device"])
    data = load_transitions(cfg)
    splits = load_splits(cfg)
    raw_dir, _ = data_paths(cfg)
    W, H = cfg["document"]["width"], cfg["document"]["height"]
    train_idx = split_indices(data, splits["train"])
    const = constant_baseline(data, train_idx)
    metrics_dir = results_subdir(cfg, "metrics")
    fig_dir = results_subdir(cfg, "figures")

    eval_docs = sorted({d for s in splits_to_eval for d in splits[s]})
    obs_maps = load_obs_maps(cfg, eval_docs)
    text_masks = {d: compute_maps(load_document(raw_dir, d).image, cfg["saliency"]).text_mask for d in eval_docs}

    models = {}
    for v in variants:
        model, norm, meta = load_checkpoint(checkpoint_path(cfg, v), device)
        models[v] = (model, norm, meta)

    results = {"seed": seed, "note": "teacher-forced one-step metrics on held-out documents; "
               "dwell in synthetic ms; expert = synthetic rule-based reader, not human data",
               "constant_baseline": {"dwell_ms": const["dwell_ms"], "displacement": const["displacement"].tolist()},
               "checkpoints": {v: {"path": str(checkpoint_path(cfg, v)), "epoch": models[v][2]["epoch"],
                                   "val_loss": models[v][2]["val_loss"]} for v in variants},
               "splits": {}}
    preds_test = {}
    for split in splits_to_eval:
        idx = split_indices(data, splits[split])
        true_d = data["dwell"][idx]
        true_m = data["displacement"][idx]
        gaze = data["gaze"][idx]
        ev = data["event"][idx]
        doc_of = data["doc_id"][idx]

        def next_in_text(pred_m):
            nxt = np.clip(gaze + pred_m, 0, 1)
            return np.array([in_text_mask(nxt[i:i + 1], text_masks[int(doc_of[i])])[0] for i in range(len(idx))])

        res = {"n_transitions": int(len(idx)), "n_documents": len(splits[split]), "doc_ids": splits[split],
               "models": {}}
        for v in variants:
            model, norm, _ = models[v]
            ds, _ = make_datasets(cfg, model.use_saliency, data, splits, obs_maps if model.use_saliency else None,
                                  which=(split,), normalizer=norm, shuffle_crops=VARIANTS[v][1])
            pm, pd = predict_dataset(model, ds[split], norm, device)
            assert np.array_equal(ds[split].idx, idx)
            if not (np.all(np.isfinite(pm)) and np.all(np.isfinite(pd)) and np.all(pd > 0)):
                raise FloatingPointError(f"invalid predictions from {v}")
            res["models"][v] = {"dwell": dwell_metrics(true_d, pd),
                                "movement": movement_metrics(true_m, pm, W, H, ev, next_in_text(pm))}
            if split == "test":
                preds_test[v] = (pm, pd)
        cm = np.repeat(const["displacement"][None], len(idx), 0)
        res["models"]["constant"] = {"dwell": dwell_metrics(true_d, np.full(len(idx), const["dwell_ms"])),
                                     "movement": movement_metrics(true_m, cm, W, H, ev, next_in_text(cm))}
        # Reference only: the expert's own noise-free dwell rule (not a model; uses hidden variables).
        res["noise_ceiling_dwell"] = dwell_metrics(true_d, data["dwell_clean"][idx])
        results["splits"][split] = res
        log(f"[{split}] n={len(idx)} " + " | ".join(
            f"{v}: r={_fmt(m['dwell']['pearson_r'])} MAE={m['dwell']['mae_ms']:.1f}ms "
            f"move={m['movement']['mean_err_px']:.1f}px" for v, m in res["models"].items())
            + f" | noise ceiling r={_fmt(res['noise_ceiling_dwell']['pearson_r'])}")

    if "test" in splits_to_eval:
        idx = split_indices(data, splits["test"])
        _save_predictions(metrics_dir / f"predictions_test_seed{seed}.csv", data, idx, preds_test, const)
        results["rollout"] = _rollouts(cfg, data, splits["test"], models, const, raw_dir, text_masks,
                                       fig_dir if make_figures else None, device)
        if make_figures:
            _figures(cfg, results, data, idx, preds_test, fig_dir)

    # File names depend on the evaluated splits, so a validation-only run can never overwrite the
    # reported test artifacts.
    tag = "" if "test" in splits_to_eval else "_" + "_".join(splits_to_eval) + "_only"
    name = f"test_metrics_seed{seed}.json" if not tag else f"metrics_seed{seed}{tag}.json"
    with open(metrics_dir / name, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    _write_summary_csv(metrics_dir / f"summary_seed{seed}{tag}.csv", results)
    save_config(cfg, metrics_dir / f"eval_config_seed{seed}{tag}.yaml")
    return results


def _fmt(r):
    return "undefined" if r is None else f"{r:.3f}"


def _save_predictions(path: Path, data, idx, preds, const):
    cols = ["doc_id", "traj_id", "t", "event", "gaze_x", "gaze_y", "expert_dx", "expert_dy", "expert_dwell_ms"]
    for v in preds:
        cols += [f"{v}_dx", f"{v}_dy", f"{v}_dwell_ms"]
    cols += ["constant_dx", "constant_dy", "constant_dwell_ms"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, i in enumerate(idx):
            row = [int(data["doc_id"][i]), int(data["traj_id"][i]), int(data["t"][i]), EVENT_NAMES[data["event"][i]],
                   *map(float, data["gaze"][i]), *map(float, data["displacement"][i]), float(data["dwell"][i])]
            for v, (pm, pd) in preds.items():
                row += [float(pm[k, 0]), float(pm[k, 1]), float(pd[k])]
            row += [float(const["displacement"][0]), float(const["displacement"][1]), const["dwell_ms"]]
            w.writerow(row)


def metrics_from_predictions_csv(path, model: str, width: int, height: int) -> dict:
    """Recompute dwell and movement metrics from a saved predictions file (for auditing)."""
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col = lambda name: np.array([float(r[name]) for r in rows])
    true_m = np.stack([col("expert_dx"), col("expert_dy")], 1)
    pred_m = np.stack([col(f"{model}_dx"), col(f"{model}_dy")], 1)
    return {"dwell": dwell_metrics(col("expert_dwell_ms"), col(f"{model}_dwell_ms")),
            "movement": movement_metrics(true_m, pred_m, width, height)}


def _rollouts(cfg, data, test_docs, models, const, raw_dir, text_masks, fig_dir, device):
    from .visualize import save_scanpath_figures

    k = cfg["evaluation"]["rollout_trajectory"]
    # The shuffled-crop control has no meaningful free-running version (its input is not tied to the
    # gaze), so it is scored teacher-forced only.
    models = {v: m for v, m in models.items() if not VARIANTS[v][1]}
    agg = {name: [] for name in ["expert", *models, "constant"]}
    per_doc = []
    for n_doc, d in enumerate(test_docs):
        doc = load_document(raw_dir, d)
        maps = compute_maps(doc.image, cfg["saliency"])
        sel = np.flatnonzero((data["doc_id"] == d) & (data["traj_id"] == k))
        sel = sel[np.argsort(data["t"][sel])]
        exp_fix, exp_dw = data["gaze"][sel].astype(np.float64), data["dwell"][sel]
        start, T = exp_fix[0], len(sel)
        row = {"doc_id": d, "n_steps": T, "expert": scanpath_metrics(exp_fix, doc, text_masks[d])}
        agg["expert"].append(row["expert"])
        paths = {}
        for v, (model, norm, _) in models.items():
            fix, dw, n_clip = rollout(Policy(model, norm, device), ReadingEnv(doc.image, cfg, maps), start, T)
            m = scanpath_metrics(fix, doc, text_masks[d])
            m.update(mean_dwell_ms=float(dw.mean()), clipped_fraction=n_clip / T)
            row[v], paths[v] = m, (fix, dw)
            agg[v].append(m)
        fix, dw, n_clip = rollout(ConstantPolicy(const["displacement"], const["dwell_ms"]),
                                  ReadingEnv(doc.image, cfg, maps), start, T)
        m = scanpath_metrics(fix, doc, text_masks[d])
        m.update(mean_dwell_ms=float(dw.mean()), clipped_fraction=n_clip / T)
        row["constant"] = m
        agg["constant"].append(m)
        per_doc.append(row)
        if fig_dir is not None and n_doc < cfg["evaluation"]["scanpath_docs"] and "full" in paths:
            save_scanpath_figures(doc.image, exp_fix, exp_dw, *paths["full"], fig_dir / f"scanpath_doc{d}",
                                  title=f"Held-out document {d}: same start fixation",
                                  max_fixations=cfg["evaluation"]["scanpath_max_fixations"])
    mean = {name: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]} for name, rows in agg.items()}
    return {"note": f"free-running rollouts from the first fixation of expert trajectory {k}, same number "
                    "of steps as that trajectory; layout used for scoring only", "mean": mean, "per_document": per_doc}


def _figures(cfg, results, data, idx, preds, fig_dir):
    from .visualize import save_ablation_figure, save_dwell_scatter, save_loss_curves

    test = results["splits"]["test"]["models"]
    for v, (_, pd) in preds.items():
        save_dwell_scatter(data["dwell"][idx], pd, test[v]["dwell"], fig_dir / f"dwell_scatter_{v}.png",
                           title=f"{MODEL_LABELS[v]}\nheld-out test documents (teacher-forced)")
    rows = [{"name": SHORT_LABELS[v], "dwell_mae_ms": test[v]["dwell"]["mae_ms"],
             "dwell_pearson_r": test[v]["dwell"]["pearson_r"], "move_err_px": test[v]["movement"]["mean_err_px"]}
            for v in MODEL_ORDER if v in test]
    save_ablation_figure(rows, fig_dir / "ablation.png")
    histories = {}
    logs = results_subdir(cfg, "logs")
    for v in preds:
        p = logs / f"train_{v}_seed{cfg['seed']}.csv"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                histories[v] = [{k: float(x) for k, x in r.items()} for r in csv.DictReader(f)]
    if histories:
        save_loss_curves(histories, fig_dir / "loss_curves.png")


def _write_summary_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "model", "n", "dwell_pearson_r", "dwell_mae_ms", "dwell_rmse_ms", "dwell_r2",
                    "move_mean_err_norm", "move_mean_err_px", "pred_next_in_text"])
        for split, res in results["splits"].items():
            for v, m in res["models"].items():
                d, mv = m["dwell"], m["movement"]
                w.writerow([split, v, d["n"], d["pearson_r"], d["mae_ms"], d["rmse_ms"], d["r2"],
                            mv["mean_err_norm"], mv["mean_err_px"], mv.get("pred_in_text_fraction")])
            c = res["noise_ceiling_dwell"]
            w.writerow([split, "noise_ceiling_reference", c["n"], c["pearson_r"], c["mae_ms"], c["rmse_ms"],
                        c["r2"], "", "", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate trained BC policies on held-out documents")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, help="seed of the checkpoints to evaluate")
    ap.add_argument("--results-dir")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"],
                    help="splits to score (use --splits val during development)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.results_dir:
        cfg["paths"]["results_dir"] = args.results_dir
    if args.checkpoint_dir:
        cfg["paths"]["checkpoint_dir"] = args.checkpoint_dir
    evaluate(cfg, splits_to_eval=tuple(args.splits))


if __name__ == "__main__":
    main()
