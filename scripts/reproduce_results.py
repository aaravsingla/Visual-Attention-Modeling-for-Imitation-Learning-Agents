"""Single command that reproduces every reported result from scratch.

    python scripts/reproduce_results.py --clean

Steps: (optionally) delete the generated data/, checkpoints/ and results/ directories,
generate documents and expert demonstrations, train the three BC variants (full, features only,
shuffled-crop control), evaluate them on the held-out test documents with baselines and rollouts,
write figures, and run the MaxEnt IRL experiment. With ``--seeds 0 1 2`` the BC part is repeated
for each training seed and a mean +- std summary is written. Wall-clock times are written to
results/logs/reproduce_timing.json.
"""

import argparse
import json
import shutil
import time

import _bootstrap  # noqa: F401

import generate_data
import run_experiments
from visual_attention.config import PROJECT_ROOT, load_config, resolve_path, results_subdir


def clean(cfg: dict) -> None:
    for name in ("data_dir", "checkpoint_dir", "results_dir"):
        p = resolve_path(cfg, name).resolve()
        if PROJECT_ROOT.resolve() not in p.parents:  # safety: never delete outside the project
            raise RuntimeError(f"refusing to delete {p}: not inside the project directory "
                               "(delete it yourself or run without --clean)")
        if p.exists():
            shutil.rmtree(p)
            print(f"removed {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--clean", action="store_true", help="delete generated data, checkpoints and results first")
    ap.add_argument("--seeds", nargs="*", type=int, help="training seeds (default: config seed)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--skip-irl", action="store_true")
    ap.add_argument("--copy-figures", action="store_true",
                    help="copy the figures shown in the README to docs/figures/")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.clean:
        clean(cfg)

    t0 = time.perf_counter()
    cfg_args = (["--config", args.config] if args.config else []) + (["--set", *args.set] if args.set else [])
    generate_data.main(cfg_args)
    t_data = time.perf_counter() - t0
    exp_args = cfg_args + (["--seeds", *map(str, args.seeds)] if args.seeds else [])
    exp_args += ["--skip-irl"] if args.skip_irl else []
    exp_timing = run_experiments.main(exp_args)
    total = time.perf_counter() - t0
    timing = {"data_generation_seconds": round(t_data, 1), "total_seconds": round(total, 1),
              "seeds": args.seeds or [cfg["seed"]],
              "bc_seconds": exp_timing.get("bc_seconds"),
              "irl_seconds": exp_timing.get("irl_seconds", "skipped")}
    with open(results_subdir(cfg, "logs") / "reproduce_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))
    if args.copy_figures:
        copy_readme_figures(cfg)


README_FIGURES = ["saliency_example.png", "expert_scanpath_example.png", "loss_curves.png",
                  "scanpath_doc0_full_rollout.png", "dwell_scatter_full.png", "ablation.png", "ablation_multiseed.png", "irl_weights.png",
                  "irl_visitation.png"]


def copy_readme_figures(cfg: dict) -> None:
    src = results_subdir(cfg, "figures")
    dst = PROJECT_ROOT / "docs" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    names = README_FIGURES + sorted(p.name for p in src.glob("scanpath_doc*_comparison.png"))[:1]
    for name in names:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            print(f"copied {name} -> docs/figures/")


if __name__ == "__main__":
    main()
