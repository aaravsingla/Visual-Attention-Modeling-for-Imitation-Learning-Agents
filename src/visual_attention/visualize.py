"""Matplotlib figures: scanpaths, saliency overlays, loss curves, dwell scatter, ablation bars.

Arrows between successive fixations are drawn as simulated saccades - a computational
analogy, not a claim about biological eye movements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .environment import to_pixels  # noqa: E402

EXPERT_COLOR = "#1f77b4"
MODEL_COLOR = "#d62728"


def draw_scanpath(image, fixations, dwell=None, ax=None, color=EXPERT_COLOR, label=None,
                  max_fixations=None, show_image=True, number=False):
    """Draw fixations (normalised coords) as circles sized by dwell, joined by saccade arrows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))
    h, w = image.shape[:2]
    fix = np.asarray(fixations, dtype=np.float64).reshape(-1, 2)
    if max_fixations is not None:
        fix = fix[:max_fixations]
        dwell = None if dwell is None else np.asarray(dwell)[:max_fixations]
    if show_image:
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    p = to_pixels(fix, w, h)
    for a, b in zip(p[:-1], p[1:]):
        if np.allclose(a, b):
            continue
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8,
                                    shrinkA=3, shrinkB=3))
    sizes = 20.0 if dwell is None else 0.25 * np.asarray(dwell, dtype=np.float64)
    ax.scatter(p[:, 0], p[:, 1], s=sizes, facecolors="none", edgecolors=color, linewidths=1.5,
               label=label, zorder=3)
    if len(p):
        ax.scatter(p[:1, 0], p[:1, 1], s=60, marker="*", color=color, zorder=4)
    if number:
        for k, (x, y) in enumerate(p):
            ax.text(x + 3, y - 3, str(k), color=color, fontsize=6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()
    return ax


def _crop_axes_to(ax, fixations_list, w, h, pad=40):
    pts = np.concatenate([to_pixels(np.asarray(f).reshape(-1, 2), w, h) for f in fixations_list])
    if len(pts) == 0:
        return
    y0, y1 = max(0, pts[:, 1].min() - pad), min(h, pts[:, 1].max() + pad)
    ax.set_ylim(y1, y0)
    ax.set_xlim(0, w)


def save_scanpath_figures(image, expert_fix, expert_dwell, model_fix, model_dwell, out_prefix,
                          title, max_fixations=40):
    """Writes ``<out_prefix>_expert.png``, ``_learned.png``, ``_comparison.png`` (first
    ``max_fixations`` fixations, cropped to them) and ``_full_rollout.png`` (whole page, all fixations)."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    full_fix, full_dwell = np.asarray(model_fix), np.asarray(model_dwell)
    ef, ed = np.asarray(expert_fix)[:max_fixations], np.asarray(expert_dwell)[:max_fixations]
    mf, md = np.asarray(model_fix)[:max_fixations], np.asarray(model_dwell)[:max_fixations]
    paths = []
    for name, items in (("expert", [(ef, ed, EXPERT_COLOR, "synthetic expert")]),
                        ("learned", [(mf, md, MODEL_COLOR, "learned policy (free-running rollout)")]),
                        ("comparison", [(ef, ed, EXPERT_COLOR, "synthetic expert"),
                                        (mf, md, MODEL_COLOR, "learned policy (free-running rollout)")])):
        fig, ax = plt.subplots(figsize=(7, 6))
        for k, (f, d, c, lab) in enumerate(items):
            draw_scanpath(image, f, d, ax=ax, color=c, label=lab, show_image=(k == 0))
        _crop_axes_to(ax, [it[0] for it in items], w, h)
        ax.set_title(f"{title}\nfirst {max_fixations} fixations; circle area ~ dwell; star = start",
                     fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        path = Path(f"{out_prefix}_{name}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 7.5))
    for ax, (f, d, c, lab) in zip(axes, ((expert_fix, expert_dwell, EXPERT_COLOR, "synthetic expert"),
                                          (full_fix, full_dwell, MODEL_COLOR, "learned policy (free-running)"))):
        draw_scanpath(image, f, d, ax=ax, color=c)
        f = np.asarray(f)
        on_edge = np.any((f <= 0) | (f >= 1), axis=1)
        ax.set_title(f"{lab}: all {len(f)} fixations\n{int(on_edge.sum())} on the page border", fontsize=9)
    fig.suptitle(f"{title}: complete scanpaths on the whole page", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = Path(f"{out_prefix}_full_rollout.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
    return paths


def save_saliency_figure(image, maps, path, title=""):
    from .saliency import overlay

    fig, axes = plt.subplots(1, 4, figsize=(16, 5.6))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("document")
    overlay(image, maps.saliency, ax=axes[1], title="saliency overlay (proxy, not human gaze)")
    im = axes[2].imshow(maps.density, cmap="viridis", vmin=0, vmax=max(float(maps.density.max()), 1e-6))
    axes[2].set_title("local ink density")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    axes[3].imshow(maps.text_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("text-region mask")
    for a in axes:
        a.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_loss_curves(histories: dict[str, list[dict]], path):
    """``histories[name]`` is a list of per-epoch dicts with train/val loss entries."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = [("loss", "total loss"), ("movement_loss", "movement loss (z-scored MSE)"),
            ("dwell_loss", "dwell loss (z-scored log-dwell MSE)")]
    for ax, (k, title) in zip(axes, keys):
        for (name, hist), color in zip(histories.items(), (MODEL_COLOR, "#ff7f0e", EXPERT_COLOR)):
            ep = [h["epoch"] for h in hist]
            ax.plot(ep, [h[f"train_{k}"] for h in hist], color=color, label=f"{name} train")
            ax.plot(ep, [h[f"val_{k}"] for h in hist], "--", color=color, label=f"{name} val")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_dwell_scatter(expert_ms, pred_ms, metrics: dict, path, title="Held-out test set"):
    expert_ms, pred_ms = np.asarray(expert_ms), np.asarray(pred_ms)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(expert_ms, pred_ms, s=4, alpha=0.25, color=MODEL_COLOR, edgecolors="none")
    lo = float(min(expert_ms.min(), pred_ms.min()))
    hi = float(max(expert_ms.max(), pred_ms.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    r = metrics.get("pearson_r")
    r_txt = "undefined" if r is None else f"{r:.3f}"
    ax.text(0.03, 0.97, f"Pearson r = {r_txt}\nMAE = {metrics['mae_ms']:.1f} ms\n"
            f"RMSE = {metrics['rmse_ms']:.1f} ms\nn = {metrics['n']}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.set_xlabel("expert dwell (synthetic ms)")
    ax.set_ylabel("predicted dwell (synthetic ms)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_ablation_figure(rows: list[dict], path, title="Held-out test set (teacher-forced, one step ahead)"):
    """``rows``: dicts with keys name, dwell_mae_ms, dwell_pearson_r, move_err_px (optional *_std)."""
    names = [r["name"] for r in rows]
    panels = [("dwell_mae_ms", "dwell MAE (ms), lower is better"),
              ("dwell_pearson_r", "dwell Pearson r, higher is better"),
              ("move_err_px", "next-fixation error (px), lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["#999999", "#ff7f0e", "#9467bd", MODEL_COLOR, "#2ca02c"]
    for ax, (k, panel_title) in zip(axes, panels):
        vals = [np.nan if r.get(k) is None else r[k] for r in rows]
        errs = [r.get(f"{k}_std", 0.0) or 0.0 for r in rows]
        has_err = any(e > 0 for e in errs)
        bars = ax.bar(range(len(rows)), vals, yerr=errs if has_err else None, capsize=4,
                      color=colors[:len(rows)])
        top = np.nanmax(np.array(vals, dtype=float) + np.array(errs)) if not np.all(np.isnan(vals)) else 1.0
        for b, v, e in zip(bars, vals, errs):
            if np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, 0.3 * top, "undefined\n(constant prediction)",
                        ha="center", va="bottom", fontsize=7, color="0.3")
                continue
            txt = f"{v:.3f}" if k == "dwell_pearson_r" else f"{v:.1f}"
            ax.text(b.get_x() + b.get_width() / 2, v + e + 0.01 * top, txt, ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.12 * top)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(names, rotation=15, fontsize=8)
        ax.set_title(panel_title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
