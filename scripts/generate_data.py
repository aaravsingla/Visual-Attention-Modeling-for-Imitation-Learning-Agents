"""Stage 1 of the pipeline: synthetic documents + expert demonstrations + document-level split.

Usage:
    python scripts/generate_data.py [--config configs/default.yaml] [--set key.sub=value ...]
"""

import argparse
import json
import time

import _bootstrap  # noqa: F401

from visual_attention.config import load_config, results_subdir, save_config
from visual_attention.dataset import data_paths, generate_dataset, load_splits, load_transitions
from visual_attention.document_generator import load_document
from visual_attention.saliency import compute_maps
from visual_attention.visualize import draw_scanpath, save_saliency_figure


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)

    t0 = time.perf_counter()
    summary = generate_dataset(cfg)
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    _, proc = data_paths(cfg)
    save_config(cfg, proc / "config.yaml")
    print(json.dumps(summary, indent=2))

    # Example figures: saliency maps and one expert scanpath of the first training document.
    import matplotlib.pyplot as plt

    fig_dir = results_subdir(cfg, "figures")
    raw, _ = data_paths(cfg)
    doc_id = load_splits(cfg)["train"][0]
    doc = load_document(raw, doc_id)
    save_saliency_figure(doc.image, compute_maps(doc.image, cfg["saliency"]),
                         fig_dir / "saliency_example.png", title=f"document {doc_id}")
    data = load_transitions(cfg)
    sel = (data["doc_id"] == doc_id) & (data["traj_id"] == 0)
    fig, ax = plt.subplots(figsize=(6, 8))
    draw_scanpath(doc.image, data["gaze"][sel][:60], data["dwell"][sel][:60], ax=ax, label="synthetic expert",
                  number=False)
    ax.set_title(f"Synthetic expert scanpath, document {doc_id}, trajectory 0 (first 60 fixations)\n"
                 "circle area ~ dwell; arrows = simulated saccades", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "expert_scanpath_example.png", dpi=130)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    main()
