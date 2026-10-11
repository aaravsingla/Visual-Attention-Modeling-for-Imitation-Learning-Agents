"""Development helper: train BC variants with config overrides and score the VALIDATION split only.

This is the procedure that produced docs/development_log.md (it was first run from a scratch copy
of this script, writing outside the repository). It never computes test-set numbers.

Usage:
    python scripts/dev_validation_run.py --out scratch/dev_C --set document.n_documents=100 expert.trajectories_per_document=5
"""

import argparse

import _bootstrap  # noqa: F401

from visual_attention.config import load_config
from visual_attention.dataset import generate_dataset
from visual_attention.evaluate import evaluate
from visual_attention.train import train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory for data, checkpoints and results")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="+", default=["full", "no_saliency"])
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    cfg["paths"] = {"data_dir": f"{args.out}/data", "checkpoint_dir": f"{args.out}/ckpt",
                    "results_dir": f"{args.out}/results"}
    generate_dataset(cfg)
    for v in args.variants:
        s = train(cfg, v, log=lambda *a: None)
        print(v, "best epoch", s["best_epoch"], "val", round(s["best_val_loss"], 4), "sec", s["train_seconds"], flush=True)
    evaluate(cfg, variants=tuple(args.variants), splits_to_eval=("val",), make_figures=False)


if __name__ == "__main__":
    main()
