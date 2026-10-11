import json

import numpy as np
import pytest

from visual_attention.dataset import (Normalizer, TransitionDataset, assert_disjoint, data_paths,
                                      generate_dataset, load_obs_maps, load_splits, load_transitions,
                                      split_documents, split_indices)
from visual_attention.document_generator import load_document
from visual_attention.environment import FEATURE_NAMES, local_features
from visual_attention.saliency import compute_maps


@pytest.fixture()
def generated(small_cfg):
    summary = generate_dataset(small_cfg)
    return small_cfg, summary


def test_split_disjoint_and_complete():
    fr = {"train": 0.7, "val": 0.15, "test": 0.15}
    splits = split_documents(range(60), fr, seed=0)
    assert_disjoint(splits)
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == list(range(60))
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (42, 9, 9)
    assert splits == split_documents(range(60), fr, seed=0)
    with pytest.raises(ValueError):
        assert_disjoint({"train": [1, 2], "test": [2, 3]})


def test_generated_files_and_schema(generated):
    cfg, summary = generated
    raw, proc = data_paths(cfg)
    assert (proc / "transitions.npz").exists() and (proc / "splits.json").exists()
    assert len(list(raw.glob("doc_*.png"))) == cfg["document"]["n_documents"]
    data = load_transitions(cfg)
    n = summary["n_transitions"]
    assert data["gaze"].shape == (n, 2) and data["features"].shape == (n, len(FEATURE_NAMES))
    assert np.allclose(data["displacement"], data["next_gaze"] - data["gaze"], atol=1e-6)
    assert np.all(data["dwell"] > 0) and np.all(np.isfinite(data["dwell"]))
    assert list(data["feature_names"]) == list(FEATURE_NAMES)


def test_splits_saved_and_document_disjoint(generated):
    cfg, _ = generated
    splits = load_splits(cfg)
    data = load_transitions(cfg)
    docs = {k: set(data["doc_id"][split_indices(data, v)].tolist()) for k, v in splits.items()}
    assert not (docs["train"] & docs["val"])
    assert not (docs["train"] & docs["test"])
    assert not (docs["val"] & docs["test"])
    # every transition belongs to exactly one split, so no trajectory straddles two splits
    total = sum(len(split_indices(data, v)) for v in splits.values())
    assert total == len(data["doc_id"])


def test_reproducible(small_cfg, tmp_path):
    generate_dataset(small_cfg)
    a = load_transitions(small_cfg)
    small_cfg["paths"]["data_dir"] = str(tmp_path / "data2")
    generate_dataset(small_cfg)
    b = load_transitions(small_cfg)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_features_are_current_state_only(generated):
    """Stored features must equal features recomputed from the image at the *current* gaze."""
    cfg, _ = generated
    data = load_transitions(cfg)
    raw, _ = data_paths(cfg)
    doc = load_document(raw, int(data["doc_id"][0]))
    maps = compute_maps(doc.image, cfg["saliency"])
    rows = np.flatnonzero(data["doc_id"] == doc.doc_id)
    for i in rows:  # every row of this document, exact equality
        f = local_features(maps, data["gaze"][i], cfg["observation"])
        assert np.array_equal(f, data["features"][i]), i
        assert np.allclose(f[:2], data["gaze"][i], atol=1e-6)
        assert not np.allclose(data["gaze"][i], data["next_gaze"][i])


def test_normalizer_roundtrip(generated):
    cfg, _ = generated
    data = load_transitions(cfg)
    idx = split_indices(data, load_splits(cfg)["train"])
    norm = Normalizer.fit(data, idx)
    m = data["displacement"][idx]
    assert np.allclose(norm.inv_movement(norm.movement(m)), m, atol=1e-5)
    d = data["dwell"][idx]
    assert np.allclose(norm.inv_dwell(norm.dwell(d)), d, rtol=1e-4)
    assert np.all(norm.inv_dwell(np.array([-50.0, 0.0, 50.0])) > 0)
    back = Normalizer.from_dict(json.loads(json.dumps(norm.to_dict())))
    assert np.allclose(back.movement(m), norm.movement(m))
    zf = norm.features(data["features"][idx])
    assert np.allclose(zf.mean(0), 0, atol=1e-3)


def test_torch_dataset_items(generated):
    cfg, _ = generated
    data = load_transitions(cfg)
    splits = load_splits(cfg)
    idx = split_indices(data, splits["train"])
    norm = Normalizer.fit(data, idx)
    maps = load_obs_maps(cfg, splits["train"])
    ds = TransitionDataset(data, idx, maps, norm, cfg["observation"], use_saliency=True)
    item = ds[3]
    s = cfg["observation"]["crop_size"]
    assert item["saliency"].shape == (1, s, s)
    assert item["features"].shape == (len(FEATURE_NAMES),)
    assert item["movement"].shape == (2,) and item["dwell"].shape == (1,)
    ds2 = TransitionDataset(data, idx, None, norm, cfg["observation"], use_saliency=False)
    assert "saliency" not in ds2[3] and len(ds2) == len(idx)
