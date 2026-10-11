import numpy as np
import pytest
import torch

from visual_attention.dataset import Normalizer, generate_dataset, load_transitions, split_indices, load_splits
from visual_attention.environment import FEATURE_NAMES
from visual_attention.model import GazePolicy, Policy, bc_loss, build_model, load_checkpoint, save_checkpoint
from visual_attention.train import checkpoint_path, train

F = len(FEATURE_NAMES)


def _batch(b=8, s=84):
    torch.manual_seed(0)
    return torch.randn(b, F), torch.rand(b, 1, s, s)


@pytest.mark.parametrize("use_saliency", [True, False])
def test_forward_shapes_and_finite(cfg, use_saliency):
    model = build_model(cfg, F, "full" if use_saliency else "no_saliency")
    f, s = _batch()
    move, dwell = model(f, s if use_saliency else None)
    assert move.shape == (8, 2) and dwell.shape == (8, 1)
    assert torch.isfinite(move).all() and torch.isfinite(dwell).all()


def test_ablation_has_no_saliency_branch(cfg):
    full = build_model(cfg, F, "full")
    abl = build_model(cfg, F, "no_saliency")
    assert hasattr(full, "cnn") and not hasattr(abl, "cnn")
    assert not any(p.dim() == 4 for p in abl.parameters())  # no conv kernels at all
    f, s = _batch()
    with pytest.raises(ValueError):
        full(f, None)
    # the ablation's output cannot depend on the saliency input
    abl.eval()
    a, _ = abl(f, None)
    b, _ = abl(f, torch.zeros_like(s))
    assert torch.equal(a, b)


def test_shuffled_crop_control(small_cfg):
    """Control: same architecture as the full model; crops come from *other* transitions."""
    from visual_attention.dataset import load_obs_maps
    from visual_attention.environment import extract_crop
    from visual_attention.train import make_datasets

    full = build_model(small_cfg, F, "full")
    ctrl = build_model(small_cfg, F, "shuffled_saliency")
    assert sum(p.numel() for p in ctrl.parameters()) == sum(p.numel() for p in full.parameters())
    generate_dataset(small_cfg)
    data, splits = load_transitions(small_cfg), load_splits(small_cfg)
    maps = load_obs_maps(small_cfg, splits["train"] + splits["val"])
    real, _ = make_datasets(small_cfg, True, data, splits, maps)
    shuf, _ = make_datasets(small_cfg, True, data, splits, maps, shuffle_crops=True)
    a, b = real["train"], shuf["train"]
    assert np.array_equal(a.idx, b.idx)
    n_same = 0
    for i in range(0, len(a), max(1, len(a) // 50)):
        ia, ib = a[i], b[i]
        assert torch.equal(ia["features"], ib["features"]) and torch.equal(ia["dwell"], ib["dwell"])
        assert torch.equal(ia["movement"], ib["movement"])
        n_same += int(torch.equal(ia["saliency"], ib["saliency"]))
    assert n_same <= 2  # crops are (almost always) not the transition's own crop
    # shuffled crops are real crops of some transition of the same split
    j = b.crop_gaze[0]
    own = extract_crop(maps[int(b.crop_doc_id[0])], j, small_cfg["observation"])
    assert np.allclose(b[0]["saliency"].numpy()[0], own)
    # the permutation is fixed (same items on every call)
    assert torch.equal(b[5]["saliency"], make_datasets(small_cfg, True, data, splits, maps, shuffle_crops=True)[0]["train"][5]["saliency"])


def test_full_model_uses_saliency(cfg):
    model = build_model(cfg, F, "full").eval()
    f, s = _batch()
    a, _ = model(f, s)
    b, _ = model(f, torch.zeros_like(s))
    assert not torch.allclose(a, b)


def test_loss_finite_and_backward(cfg):
    model = build_model(cfg, F, "full")
    f, s = _batch()
    move, dwell = model(f, s)
    loss, lm, ld = bc_loss(move, dwell, torch.randn(8, 2), torch.randn(8, 1), 1.0, 0.5)
    assert torch.isfinite(loss) and torch.isclose(loss, lm + 0.5 * ld)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_optimizer_step_reduces_loss(cfg):
    torch.manual_seed(0)
    model = build_model(cfg, F, "no_saliency")
    model.eval()  # disable dropout so the comparison is deterministic
    f = torch.randn(64, F)
    tm, td = torch.randn(64, 2), torch.randn(64, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(50):
        loss, _, _ = bc_loss(*model(f), tm, td)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_checkpoint_roundtrip(cfg, tmp_path):
    model = build_model(cfg, F, "full").eval()
    norm = Normalizer(np.zeros(F), np.ones(F), [0.01, 0.0], [0.05, 0.02], 5.5, 0.2)
    save_checkpoint(tmp_path / "m.pt", model, norm, {"epoch": 3})
    back, nb, meta = load_checkpoint(tmp_path / "m.pt")
    f, s = _batch()
    assert torch.allclose(model(f, s)[0], back(f, s)[0])
    assert meta["epoch"] == 3 and np.allclose(nb.move_std, norm.move_std)


def test_policy_outputs_positive_dwell(cfg):
    model = build_model(cfg, F, "full")
    norm = Normalizer(np.zeros(F), np.ones(F), [0.01, 0.0], [0.05, 0.02], 5.5, 0.2)
    pol = Policy(model, norm)
    for scale in (0.0, 1.0, 100.0):
        obs = {"features": np.full(F, scale, np.float32), "saliency": np.ones((1, 84, 84), np.float32)}
        move, dwell = pol.act(obs)
        assert move.shape == (2,) and np.all(np.isfinite(move))
        assert np.isfinite(dwell) and dwell > 0


def test_small_training_run(small_cfg):
    generate_dataset(small_cfg)
    for variant in ("full", "no_saliency", "shuffled_saliency"):
        s = train(small_cfg, variant, log=lambda *_: None)
        assert s["epochs_run"] == 2 and np.isfinite(s["best_val_loss"])
        assert checkpoint_path(small_cfg, variant).exists()
        model, norm, meta = load_checkpoint(checkpoint_path(small_cfg, variant))
        assert meta["variant"] == variant and meta["epoch"] == s["best_epoch"]
        assert model.use_saliency == (variant != "no_saliency")
        for h in s["history"]:
            assert np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"])
    # normaliser statistics come from the training split only
    data = load_transitions(small_cfg)
    tr = split_indices(data, load_splits(small_cfg)["train"])
    assert np.isclose(norm.logdwell_mean, np.log(data["dwell"][tr].astype(np.float64)).mean(), atol=1e-6)
