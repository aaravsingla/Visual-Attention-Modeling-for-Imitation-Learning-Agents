"""Stages 2-4 of the pipeline: train the three BC variants (full, features only, shuffled-crop
control), evaluate them on held-out documents, make figures, then run the MaxEnt IRL experiment
(unless --skip-irl).

Requires data from ``scripts/generate_data.py``.

Usage:
    python scripts/run_experiments.py                         # seed from config, BC + IRL
    python scripts/run_experiments.py --seeds 0 1 2 --skip-irl # several training seeds, mean +- std
    python scripts/run_experiments.py --skip-train --skip-irl # re-evaluate existing checkpoints only
    python scripts/run_experiments.py --only-irl              # just the IRL experiment
"""

import argparse
import copy
import json
import time

import numpy as np

import _bootstrap  # noqa: F401

from visual_attention.config import load_config, results_subdir, save_config
from visual_attention.dataset import load_obs_maps, load_splits, load_transitions
from visual_attention.evaluate import evaluate
from visual_attention.train import VARIANTS, train

SUMMARY_KEYS = [("dwell", "pearson_r"), ("dwell", "mae_ms"), ("dwell", "rmse_ms"), ("dwell", "r2"),
                ("movement", "mean_err_norm"), ("movement", "mean_err_px")]


def run_bc(cfg: dict, seeds, skip_train: bool) -> dict:
    data, splits = load_transitions(cfg), load_splits(cfg)
    obs_maps = None if skip_train else load_obs_maps(cfg, splits["train"] + splits["val"])
    per_seed, timings = {}, {}
    for k, seed in enumerate(seeds):
        c = copy.deepcopy(cfg)
        c["seed"] = seed
        t0 = time.perf_counter()
        if not skip_train:
            for v in VARIANTS:
                s = train(c, v, data=data, splits=splits, obs_maps=obs_maps)
                print(f"[seed {seed}] {v}: best epoch {s['best_epoch']} / {s['epochs_run']}, "
                      f"{s['train_seconds']} s")
        res = evaluate(c, make_figures=(k == 0))
        timings[seed] = round(time.perf_counter() - t0, 1)
        per_seed[seed] = res
    return {"per_seed": per_seed, "seconds_per_seed": timings}


def aggregate(per_seed: dict) -> dict:
    """Mean and sample std over training seeds of the main held-out test metrics."""
    out = {}
    models = per_seed[next(iter(per_seed))]["splits"]["test"]["models"]
    for v in models:
        out[v] = {}
        for grp, key in SUMMARY_KEYS:
            vals = [r["splits"]["test"]["models"][v][grp][key] for r in per_seed.values()]
            vals = [x for x in vals if x is not None]
            out[v][f"{grp}_{key}"] = None if not vals else {
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals), "values": vals}
    rollout = {}
    for v in per_seed[next(iter(per_seed))]["rollout"]["mean"]:
        rollout[v] = {}
        for key in per_seed[next(iter(per_seed))]["rollout"]["mean"][v]:
            vals = [r["rollout"]["mean"][v][key] for r in per_seed.values()]
            rollout[v][key] = {"mean": float(np.mean(vals)),
                               "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return {"seeds": list(per_seed), "test": out, "rollout": rollout}


def save_multiseed(cfg: dict, agg: dict) -> None:
    from visual_attention.visualize import save_ablation_figure

    mdir = results_subdir(cfg, "metrics")
    with open(mdir / "multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    with open(mdir / "multiseed_summary.csv", "w", encoding="utf-8") as f:
        f.write("model," + ",".join(f"{g}_{k}_mean,{g}_{k}_std" for g, k in SUMMARY_KEYS) + "\n")
        for v, m in agg["test"].items():
            cells = []
            for g, k in SUMMARY_KEYS:
                x = m[f"{g}_{k}"]
                cells += ["", ""] if x is None else [f"{x['mean']:.4f}", f"{x['std']:.4f}"]
            f.write(v + "," + ",".join(cells) + "\n")
    from visual_attention.evaluate import MODEL_ORDER, SHORT_LABELS

    rows = []
    for v in [m for m in MODEL_ORDER if m in agg["test"]]:
        m = agg["test"][v]
        g = lambda key: None if m[key] is None else m[key]["mean"]
        s = lambda key: 0.0 if m[key] is None else m[key]["std"]
        rows.append({"name": SHORT_LABELS[v], "dwell_mae_ms": g("dwell_mae_ms"), "dwell_mae_ms_std": s("dwell_mae_ms"),
                     "dwell_pearson_r": g("dwell_pearson_r"), "dwell_pearson_r_std": s("dwell_pearson_r"),
                     "move_err_px": g("movement_mean_err_px"), "move_err_px_std": s("movement_mean_err_px")})
    save_ablation_figure(rows, results_subdir(cfg, "figures") / "ablation_multiseed.png",
                         title=f"Held-out test set, mean +- std over {len(agg['seeds'])} training seeds "
                               "(teacher-forced, one step ahead)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seeds", nargs="*", type=int, help="training seeds (default: config seed)")
    ap.add_argument("--skip-train", action="store_true", help="evaluate existing checkpoints only")
    ap.add_argument("--skip-irl", action="store_true")
    ap.add_argument("--only-irl", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    seeds = args.seeds or [cfg["seed"]]
    timing = {}
    save_config(cfg, results_subdir(cfg, "logs") / "run_config.yaml")

    if not args.only_irl:
        t0 = time.perf_counter()
        bc = run_bc(cfg, seeds, args.skip_train)
        timing["bc_seconds"] = round(time.perf_counter() - t0, 1)
        timing["bc_seconds_per_seed"] = bc["seconds_per_seed"]
        if len(seeds) > 1:
            save_multiseed(cfg, aggregate(bc["per_seed"]))
    if not args.skip_irl:
        from visual_attention.irl import run_irl_experiment

        t0 = time.perf_counter()
        run_irl_experiment(cfg)
        timing["irl_seconds"] = round(time.perf_counter() - t0, 1)
    name = "run_experiments_timing_seeds" + "-".join(map(str, seeds)) + ("" if args.skip_irl else "_irl") + ".json"
    with open(results_subdir(cfg, "logs") / name, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))
    return timing


if __name__ == "__main__":
    main()
