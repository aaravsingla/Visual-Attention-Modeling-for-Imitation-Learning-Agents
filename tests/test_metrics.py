import json

import numpy as np
import pytest
from scipy.stats import pearsonr

from visual_attention.config import resolve_path
from visual_attention.dataset import generate_dataset, load_splits, load_transitions, split_indices
from visual_attention.evaluate import (constant_baseline, displacement_errors, dwell_metrics, evaluate,
                                       metrics_from_predictions_csv, movement_metrics, pearson)
from visual_attention.train import train


def test_pearson_known_values():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=200), rng.normal(size=200)
    assert pearson(x, y) == pytest.approx(pearsonr(x, y)[0], abs=1e-10)


def test_pearson_degenerate_cases():
    assert pearson([5, 5, 5], [1, 2, 3]) is None
    assert pearson([1, 2, 3], [7, 7, 7]) is None
    assert pearson([1], [2]) is None
    m = dwell_metrics([200, 200, 200], [190, 210, 200])
    assert m["pearson_r"] is None and m["r2"] is None
    assert m["mae_ms"] == pytest.approx(20 / 3)


def test_mae_rmse_in_ms():
    m = dwell_metrics([100, 200, 300, 400], [110, 190, 330, 400])
    assert m["mae_ms"] == pytest.approx((10 + 10 + 30 + 0) / 4)
    assert m["rmse_ms"] == pytest.approx(np.sqrt((100 + 100 + 900 + 0) / 4))
    assert m["n"] == 4


def test_displacement_errors_units():
    true = np.array([[0.0, 0.0], [0.1, 0.0]])
    pred = np.array([[0.03, 0.04], [0.1, 0.01]])
    norm, px = displacement_errors(true, pred, width=600, height=800)
    assert norm == pytest.approx([0.05, 0.01])
    assert px == pytest.approx([np.hypot(18, 32), 8.0])
    m = movement_metrics(true, pred, 600, 800, events=[0, 3])
    assert m["mean_err_px"] == pytest.approx(np.mean(px))
    assert m["by_event_mean_err_px"]["return_sweep"] == pytest.approx(8.0)
    assert m["by_event_mean_err_px"]["skip"] is None


def test_constant_baseline_uses_training_labels_only():
    data = {"dwell": np.array([100.0, 200.0, 9000.0]), "displacement": np.array([[0.1, 0], [0.3, 0], [5.0, 5.0]])}
    c = constant_baseline(data, np.array([0, 1]))
    assert c["dwell_ms"] == pytest.approx(150.0)
    assert np.allclose(c["displacement"], [0.2, 0.0])


@pytest.fixture()
def evaluated(small_cfg):
    generate_dataset(small_cfg)
    for v in ("full", "no_saliency", "shuffled_saliency"):
        train(small_cfg, v, log=lambda *_: None)
    res = evaluate(small_cfg, log=lambda *_: None)
    return small_cfg, res


def test_evaluation_outputs(evaluated):
    cfg, res = evaluated
    out = resolve_path(cfg, "results_dir")
    seed = cfg["seed"]
    for p in (f"metrics/test_metrics_seed{seed}.json", f"metrics/predictions_test_seed{seed}.csv",
              f"metrics/summary_seed{seed}.csv", "figures/dwell_scatter_full.png", "figures/ablation.png",
              "figures/loss_curves.png"):
        assert (out / p).exists(), p
    assert list((out / "figures").glob("scanpath_doc*_comparison.png"))
    with open(out / f"metrics/test_metrics_seed{seed}.json") as f:
        saved = json.load(f)
    test = saved["splits"]["test"]
    data = load_transitions(cfg)
    splits = load_splits(cfg)
    assert test["n_transitions"] == len(split_indices(data, splits["test"]))
    assert test["doc_ids"] == splits["test"]
    for v in ("full", "no_saliency", "constant"):
        assert test["models"][v]["dwell"]["n"] == test["n_transitions"]
    assert test["models"]["constant"]["dwell"]["pearson_r"] is None  # constant prediction: undefined


def test_validation_only_run_keeps_test_artifacts(evaluated):
    cfg, _ = evaluated
    out = resolve_path(cfg, "results_dir") / "metrics"
    seed = cfg["seed"]
    before = (out / f"test_metrics_seed{seed}.json").read_bytes()
    res = evaluate(cfg, splits_to_eval=("val",), make_figures=False, log=lambda *_: None)
    assert list(res["splits"]) == ["val"]
    assert (out / f"test_metrics_seed{seed}.json").read_bytes() == before
    assert (out / f"metrics_seed{seed}_val_only.json").exists()


def test_metrics_reproducible_from_saved_predictions(evaluated):
    cfg, res = evaluated
    out = resolve_path(cfg, "results_dir")
    path = out / f"metrics/predictions_test_seed{cfg['seed']}.csv"
    W, H = cfg["document"]["width"], cfg["document"]["height"]
    for v in ("full", "no_saliency", "shuffled_saliency"):
        re = metrics_from_predictions_csv(path, v, W, H)
        saved = res["splits"]["test"]["models"][v]
        assert re["dwell"]["pearson_r"] == pytest.approx(saved["dwell"]["pearson_r"], abs=1e-6)
        assert re["dwell"]["mae_ms"] == pytest.approx(saved["dwell"]["mae_ms"], rel=1e-5)
        assert re["movement"]["mean_err_px"] == pytest.approx(saved["movement"]["mean_err_px"], rel=1e-5)


def test_constant_baseline_independent_of_test_labels(small_cfg):
    """Changing test-set labels must not change the constant baseline."""
    generate_dataset(small_cfg)
    data = load_transitions(small_cfg)
    splits = load_splits(small_cfg)
    tr = split_indices(data, splits["train"])
    te = split_indices(data, splits["test"])
    a = constant_baseline(data, tr)
    data["dwell"][te] *= 10
    data["displacement"][te] += 1.0
    b = constant_baseline(data, tr)
    assert a["dwell_ms"] == b["dwell_ms"] and np.array_equal(a["displacement"], b["displacement"])


def test_rollout_uses_policy_outputs_only(cfg):
    """Free-running rollout: each new gaze is the previous gaze plus the policy's own displacement."""
    from visual_attention.document_generator import generate_document
    from visual_attention.environment import ReadingEnv
    from visual_attention.evaluate import ConstantPolicy, rollout, scanpath_metrics

    class Recorder:
        def __init__(self):
            self.seen = []

        def act(self, obs):
            self.seen.append(obs["features"][:2].copy())
            return np.array([0.02, 0.005]), 210.0

    doc = generate_document(0, cfg["document"], seed=0)
    pol = Recorder()
    fix, dwell, n_clip = rollout(pol, ReadingEnv(doc.image, cfg), [0.1, 0.1], 10)
    assert fix.shape == (10, 2) and np.all(dwell == 210.0) and n_clip == 0
    assert np.allclose(np.diff(fix, axis=0), [0.02, 0.005])
    assert np.allclose(np.array(pol.seen), fix, atol=1e-6)  # the policy saw its own positions
    fix2, _, _ = rollout(ConstantPolicy([0.0, 0.3], 100.0), ReadingEnv(doc.image, cfg), [0.5, 0.5], 4)
    assert np.allclose(fix2[:, 1], [0.5, 0.8, 1.0, 1.0])  # clipped at the page boundary
    m = scanpath_metrics(fix, doc, np.zeros((doc.height, doc.width), np.float32))
    assert m["in_text_fraction"] == 0.0 and 0 <= m["word_coverage"] <= 1
