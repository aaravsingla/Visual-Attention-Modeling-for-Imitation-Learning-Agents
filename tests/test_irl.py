import itertools
import json

import numpy as np
import pytest

from visual_attention.config import resolve_path
from visual_attention.irl import (ACTION_NAMES, FEATURE_NAMES, build_mdp, empirical_features, feature_expectations,
                                  fit_logit_bc, irl_objective_and_grad, logit_policy, maxent_irl, mean_log_likelihood,
                                  run_irl_experiment, sample_demos, sample_trajectories, soft_value_iteration,
                                  state_visitation)

W_TRUE = np.array([1.0, 2.0, -1.5, -0.5, -1.0])


def _mdp(rows=3, cols=4, horizon=6, seed=0):
    rng = np.random.default_rng(seed)
    return build_mdp(rng.random(rows * cols), rng.random(rows * cols), rows, cols, horizon)


def test_shapes_and_validity():
    m = _mdp()
    S, A, K = 12, len(ACTION_NAMES), len(FEATURE_NAMES)
    assert m.phi.shape == (S, A, K) and m.valid.shape == (S, A) and m.reward(W_TRUE).shape == (S, A)
    assert np.all(m.valid.sum(1) >= 1)  # every state has at least one legal action
    assert not m.valid[3, 0] and not m.valid[0, 2] and not m.valid[11, 3]  # grid edges
    assert m.next_state[5, 3] == 8  # sweep goes to the first cell of the next row
    pi, V = soft_value_iteration(m, W_TRUE)
    assert pi.shape == (6, S, A) and V.shape == (7, S)
    assert np.allclose(pi.sum(-1), 1.0) and np.all(pi[:, ~m.valid] == 0) and np.all(pi >= 0)
    assert np.all(np.isfinite(V))


def test_soft_value_matches_brute_force():
    """V_0(s0) must equal log sum over all valid action sequences of exp(total reward)."""
    m = _mdp(rows=2, cols=3, horizon=4)
    r = m.reward(W_TRUE)
    total = []
    for seq in itertools.product(range(len(ACTION_NAMES)), repeat=m.horizon):
        s, ret, ok = 0, 0.0, True
        for a in seq:
            if not m.valid[s, a]:
                ok = False
                break
            ret += r[s, a]
            s = m.next_state[s, a]
        if ok:
            total.append(ret)
    _, V = soft_value_iteration(m, W_TRUE)
    assert V[0, 0] == pytest.approx(np.log(np.sum(np.exp(total))), rel=1e-10)


def test_feature_expectations_match_monte_carlo():
    m = _mdp()
    pi, _ = soft_value_iteration(m, W_TRUE)
    exact = feature_expectations(m, pi)
    S_, A_ = sample_trajectories(m, pi, 4000, np.random.default_rng(0))
    mc = empirical_features(m, S_, A_)
    assert np.allclose(exact, mc, atol=0.08)
    D = state_visitation(m, pi)
    assert np.allclose(D.sum(1), 1.0)


def test_gradient_matches_finite_differences():
    mdps = [_mdp(seed=s) for s in range(3)]
    demos = sample_demos(mdps, W_TRUE, 30, seed=0, salt=1)
    w = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
    _, g = irl_objective_and_grad(mdps, demos, w)
    eps = 1e-5
    for k in range(len(w)):
        e = np.zeros(len(w))
        e[k] = eps
        num = (irl_objective_and_grad(mdps, demos, w + e)[0] - irl_objective_and_grad(mdps, demos, w - e)[0]) / (2 * eps)
        assert g[k] == pytest.approx(num, abs=1e-6)


def test_irl_recovers_known_reward_and_beats_myopic_bc():
    mdps = [_mdp(rows=4, cols=5, horizon=10, seed=s) for s in range(12)]
    demos = sample_demos(mdps, W_TRUE, 200, seed=0, salt=1)
    w, hist = maxent_irl(mdps, demos, n_iters=400, lr=0.05)
    assert all(np.isfinite(h["log_likelihood"]) for h in hist)
    assert hist[-1]["log_likelihood"] > hist[0]["log_likelihood"]
    cos = w @ W_TRUE / np.linalg.norm(w) / np.linalg.norm(W_TRUE)
    assert cos > 0.98 and np.linalg.norm(w - W_TRUE) / np.linalg.norm(W_TRUE) < 0.2
    # learned policy reproduces expert feature expectations
    for m, (S_, A_) in zip(mdps[:3], demos[:3]):
        assert np.allclose(feature_expectations(m, soft_value_iteration(m, w)[0]), empirical_features(m, S_, A_), atol=0.6)
    # held-out documents: IRL policy is at least as likely as one-step BC on new documents
    test = [_mdp(rows=4, cols=5, horizon=10, seed=100 + s) for s in range(5)]
    tdem = sample_demos(test, W_TRUE, 100, seed=0, salt=2)
    th = fit_logit_bc(mdps, demos, n_iters=400, lr=0.05)
    nll_irl = np.mean([-mean_log_likelihood(soft_value_iteration(m, w)[0], S_, A_) for m, (S_, A_) in zip(test, tdem)])
    nll_bc = np.mean([-mean_log_likelihood(logit_policy(m, th), S_, A_) for m, (S_, A_) in zip(test, tdem)])
    assert nll_irl <= nll_bc + 1e-3


def test_logit_policy_valid():
    m = _mdp()
    pi = logit_policy(m, np.array([5.0, -3.0, 2.0, 0.0, 1.0]))
    assert np.allclose(pi.sum(1), 1) and np.all(pi[~m.valid] == 0)


def test_experiment_outputs(small_cfg):
    c = small_cfg
    c["irl"].update(n_train_docs=3, n_test_docs=2, trajectories_per_doc=5, n_iters=30, sample_sizes=[1, 5])
    res = run_irl_experiment(c, log=lambda *_: None)
    out = resolve_path(c, "results_dir")
    for p in ("metrics/irl_results.json", "figures/irl_weights.png", "figures/irl_visitation.png",
              "logs/irl_optimization.csv"):
        assert (out / p).exists(), p
    saved = json.loads((out / "metrics/irl_results.json").read_text())
    assert len(saved["w_irl"]) == len(FEATURE_NAMES) and np.all(np.isfinite(saved["w_irl"]))
    assert set(saved["heldout"]) >= {"true_reward_soft_optimal", "irl_soft_optimal", "bc_logit_one_step",
                                     "uniform_valid_actions"}
    assert saved["heldout"]["true_reward_soft_optimal"]["tv_distance_to_true_policy"] == pytest.approx(0.0, abs=1e-12)
    assert set(saved["train_doc_ids"]).isdisjoint(saved["test_doc_ids"])
    assert len(res["sample_size_sweep"]) == 2
