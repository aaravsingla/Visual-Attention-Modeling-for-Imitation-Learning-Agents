"""Maximum-entropy inverse reinforcement learning on a small discretized reading MDP.

This is a separate, controlled experiment. It is **not** connected to the Behavior Cloning network.

MDP (one per synthetic document)
--------------------------------
* States: cells of an ``R x C`` grid laid over the page area inside the margins, in reading order
  ``s = row * C + col``. The episode starts in cell (0, 0) and lasts exactly ``T`` steps.
* Actions: ``right`` (col + 1), ``skip`` (col + 2), ``regress`` (col - 1) and ``sweep`` (first
  cell of the next row). Moves that would leave the grid are invalid (masked out), so dynamics are
  deterministic.
* Features ``phi(s, a)`` (``FEATURE_NAMES``): saliency and ink density of the landing cell
  (per-document min-max scaled to [0, 1], from the same handcrafted maps as P0), and indicators for
  regress / skip / sweep (``right`` is the reference action).
* Reward ``R(s, a) = w . phi(s, a)``.

Ground truth and demonstrations
-------------------------------
The expert is the finite-horizon *soft-optimal* (MaxEnt) policy for a **known** weight vector
``w_true`` (config ``irl.true_weights``)::

    Q_t(s, a) = R(s, a) + V_{t+1}(s'(s, a)),   V_t(s) = log sum_a exp Q_t(s, a),   V_T = 0
    pi_t(a | s) = exp(Q_t(s, a) - V_t(s))

Demonstrations are sampled from this policy. Because the expert really is a MaxEnt agent with a
linear reward in the model class, recovering ``w_true`` is a well-posed test of the IRL code.
This is recovery of a *synthetic expert's* reward, not inference of the reward used by human readers.

IRL
---
Maximise the average log-likelihood of the demonstrations under the soft-optimal policy for ``w``.
For deterministic dynamics, ``log P(tau | w) = w . Phi(tau) - V_0(s_0)``, so the gradient is exactly

    grad = mean_tau Phi(tau) - E_{pi_w}[Phi]      (expert minus learner feature expectations)

where ``E_{pi_w}[Phi]`` is computed exactly by a forward pass over state-visitation frequencies.
The objective is concave in ``w``. Updates use Adam (gradient ascent). The derived policy is the
soft-optimal policy for the learned ``w``.

A one-step *behavior cloning* baseline on the same features (conditional logit
``pi(a | s) ∝ exp(theta . phi(s, a))``, fitted by maximum likelihood, no planning) is included for
comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from .config import results_subdir, save_config
from .document_generator import generate_document
from .saliency import compute_maps, downsample

FEATURE_NAMES = ("landing_saliency", "landing_density", "regress", "skip", "sweep")
ACTION_NAMES = ("right", "skip", "regress", "sweep")
NEG_INF = -1e30


@dataclass
class GridMDP:
    rows: int
    cols: int
    horizon: int
    next_state: np.ndarray   # (S, A) int, -1 where invalid
    valid: np.ndarray        # (S, A) bool
    phi: np.ndarray          # (S, A, K) float
    saliency: np.ndarray     # (S,) cell saliency in [0, 1]
    density: np.ndarray      # (S,) cell density in [0, 1]

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    def reward(self, w) -> np.ndarray:
        return self.phi @ np.asarray(w, dtype=np.float64)


def _scale01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def cell_features(maps, rows: int, cols: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean saliency and density per grid cell (area average), each min-max scaled to [0, 1]."""
    H, W = maps.shape
    sal = downsample(maps.saliency[margin:H - margin, margin:W - margin], (rows, cols)).ravel()
    den = downsample(maps.density[margin:H - margin, margin:W - margin], (rows, cols)).ravel()
    return _scale01(sal.astype(np.float64)), _scale01(den.astype(np.float64))


def build_mdp(saliency: np.ndarray, density: np.ndarray, rows: int, cols: int, horizon: int) -> GridMDP:
    S, A, K = rows * cols, len(ACTION_NAMES), len(FEATURE_NAMES)
    nxt = -np.ones((S, A), dtype=int)
    for s in range(S):
        r, c = divmod(s, cols)
        if c + 1 < cols:
            nxt[s, 0] = s + 1
        if c + 2 < cols:
            nxt[s, 1] = s + 2
        if c - 1 >= 0:
            nxt[s, 2] = s - 1
        if r + 1 < rows:
            nxt[s, 3] = (r + 1) * cols
    valid = nxt >= 0
    phi = np.zeros((S, A, K))
    for s in range(S):
        for a in range(A):
            if valid[s, a]:
                s2 = nxt[s, a]
                phi[s, a] = [saliency[s2], density[s2], a == 2, a == 1, a == 3]
    return GridMDP(rows, cols, horizon, nxt, valid, phi, saliency, density)


def soft_value_iteration(mdp: GridMDP, w) -> tuple[np.ndarray, np.ndarray]:
    """Finite-horizon soft Bellman backup. Returns policy ``(T, S, A)`` and values ``(T+1, S)``."""
    T, S = mdp.horizon, mdp.n_states
    r = np.where(mdp.valid, mdp.reward(w), NEG_INF)
    V = np.zeros((T + 1, S))
    pi = np.zeros((T, S, len(ACTION_NAMES)))
    safe_next = np.where(mdp.valid, mdp.next_state, 0)
    for t in range(T - 1, -1, -1):
        Q = np.where(mdp.valid, r + V[t + 1][safe_next], NEG_INF)
        m = Q.max(1, keepdims=True)
        V[t] = (m + np.log(np.exp(Q - m).sum(1, keepdims=True)))[:, 0]
        pi[t] = np.exp(Q - V[t][:, None]) * mdp.valid
    return pi, V


def state_visitation(mdp: GridMDP, pi: np.ndarray, start: int = 0) -> np.ndarray:
    """``D[t, s]`` = probability of being in ``s`` at step t. ``pi`` is ``(T, S, A)`` or ``(S, A)``."""
    T, S = mdp.horizon, mdp.n_states
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    D = np.zeros((T, S))
    D[0, start] = 1.0
    safe_next = np.where(mdp.valid, mdp.next_state, 0)
    for t in range(T - 1):
        flow = D[t][:, None] * pi[t]
        np.add.at(D[t + 1], safe_next.ravel(), flow.ravel())
    return D


def feature_expectations(mdp: GridMDP, pi: np.ndarray, start: int = 0) -> np.ndarray:
    """Exact expected feature sum over the horizon, ``E_pi[sum_t phi(s_t, a_t)]``."""
    T = mdp.horizon
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    D = state_visitation(mdp, pi, start)
    return np.einsum("ts,tsa,sak->k", D, pi, mdp.phi)


def sample_trajectories(mdp: GridMDP, pi: np.ndarray, n: int, rng: np.random.Generator,
                        start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` trajectories of ``T`` (state, action) pairs from a time-indexed policy ``(T, S, A)``."""
    T = mdp.horizon
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    states = np.zeros((n, T), dtype=int)
    actions = np.zeros((n, T), dtype=int)
    for i in range(n):
        s = start
        for t in range(T):
            p = pi[t, s]
            a = int(rng.choice(len(p), p=p / p.sum()))
            states[i, t], actions[i, t] = s, a
            s = int(mdp.next_state[s, a])
    return states, actions


def empirical_features(mdp: GridMDP, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return mdp.phi[states, actions].sum(1).mean(0)


def mean_log_likelihood(pi: np.ndarray, states: np.ndarray, actions: np.ndarray) -> float:
    """Average per-step log-probability of demonstrated actions. ``pi``: ``(T, S, A)`` or ``(S, A)``."""
    T = states.shape[1]
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    p = pi[np.arange(T)[None, :], states, actions]
    return float(np.log(np.maximum(p, 1e-300)).mean())


def uniform_policy(mdp: GridMDP) -> np.ndarray:
    return mdp.valid / mdp.valid.sum(1, keepdims=True)


class Adam:
    def __init__(self, dim: int, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        """Returns the ascent step for ``grad``."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh, vh = self.m / (1 - self.b1 ** self.t), self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def irl_objective_and_grad(mdps, demos, w, l2: float = 0.0):
    """Mean (over documents) per-trajectory log-likelihood and its exact gradient in ``w``."""
    ll, grad = 0.0, np.zeros(len(w))
    for mdp, (S_, A_) in zip(mdps, demos):
        pi, V = soft_value_iteration(mdp, w)
        ll += mean_log_likelihood(pi, S_, A_) * mdp.horizon
        grad += empirical_features(mdp, S_, A_) - feature_expectations(mdp, pi)
    n = len(mdps)
    return ll / n - 0.5 * l2 * float(w @ w), grad / n - l2 * w


def maxent_irl(mdps, demos, n_iters: int, lr: float, l2: float = 0.0, w0=None, tol: float = 1e-6):
    """Gradient ascent (Adam) on the MaxEnt log-likelihood. Returns ``w`` and a history list."""
    K = mdps[0].phi.shape[-1]
    w = np.zeros(K) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    opt = Adam(K, lr)
    history = []
    for it in range(n_iters):
        obj, g = irl_objective_and_grad(mdps, demos, w, l2)
        if not (np.isfinite(obj) and np.all(np.isfinite(g))):
            raise FloatingPointError("non-finite IRL objective or gradient")
        history.append({"iter": it, "log_likelihood": obj, "grad_norm": float(np.linalg.norm(g)),
                        "w": w.tolist()})
        if np.linalg.norm(g) < tol:
            break
        w = w + opt.step(g)
    return w, history


def logit_policy(mdp: GridMDP, theta) -> np.ndarray:
    """Myopic conditional-logit policy ``(S, A)``: ``pi(a|s) ∝ exp(theta . phi(s, a))`` over valid actions."""
    z = np.where(mdp.valid, mdp.reward(theta), NEG_INF)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z) * mdp.valid
    return e / e.sum(1, keepdims=True)


def fit_logit_bc(mdps, demos, n_iters: int, lr: float):
    """One-step behavior cloning on the IRL features (no planning). Maximum likelihood with Adam."""
    K = mdps[0].phi.shape[-1]
    th = np.zeros(K)
    opt = Adam(K, lr)
    for _ in range(n_iters):
        g = np.zeros(K)
        for mdp, (S_, A_) in zip(mdps, demos):
            pi = logit_policy(mdp, th)
            taken = mdp.phi[S_, A_]                                   # (n, T, K)
            expected = np.einsum("nta,ntak->ntk", pi[S_], mdp.phi[S_])
            g += (taken - expected).mean((0, 1))
        g /= len(mdps)
        if np.linalg.norm(g) < 1e-7:
            break
        th = th + opt.step(g)
    return th


def policy_comparison(mdp: GridMDP, pi_ref: np.ndarray, pi: np.ndarray) -> dict:
    """Visitation-weighted total-variation distance and argmax agreement with the reference policy."""
    T = mdp.horizon
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    D = state_visitation(mdp, pi_ref)
    tv = 0.5 * np.abs(pi - pi_ref).sum(-1)                           # (T, S)
    agree = pi.argmax(-1) == pi_ref.argmax(-1)
    return {"tv_distance": float((D * tv).sum() / D.sum()), "argmax_agreement": float((D * agree).sum() / D.sum())}


def expected_true_return(mdp: GridMDP, pi: np.ndarray, w_true) -> float:
    """Expected sum of ground-truth rewards (no entropy term) when following ``pi``."""
    T = mdp.horizon
    pi = np.broadcast_to(pi, (T,) + pi.shape[-2:])
    return float(feature_expectations(mdp, pi) @ np.asarray(w_true))


def make_mdps(cfg: dict, doc_ids) -> tuple[list[GridMDP], list]:
    ic = cfg["irl"]
    out, docs = [], []
    for d in doc_ids:
        doc = generate_document(int(d), cfg["document"], cfg.get("data_seed", cfg["seed"]))
        maps = compute_maps(doc.image, cfg["saliency"])
        sal, den = cell_features(maps, ic["grid_rows"], ic["grid_cols"], cfg["document"]["margin"])
        out.append(build_mdp(sal, den, ic["grid_rows"], ic["grid_cols"], ic["horizon"]))
        docs.append(doc)
    return out, docs


def sample_demos(mdps, w_true, n_per_doc: int, seed: int, salt: int):
    demos = []
    for k, mdp in enumerate(mdps):
        pi, _ = soft_value_iteration(mdp, w_true)
        demos.append(sample_trajectories(mdp, pi, n_per_doc, np.random.default_rng([seed, salt, k])))
    return demos


def recovery_metrics(w_hat, w_true) -> dict:
    w_hat, w_true = np.asarray(w_hat, dtype=np.float64), np.asarray(w_true, dtype=np.float64)
    cos = float(w_hat @ w_true / (np.linalg.norm(w_hat) * np.linalg.norm(w_true) + 1e-12))
    return {"cosine_similarity": cos, "relative_l2_error": float(np.linalg.norm(w_hat - w_true) / np.linalg.norm(w_true)),
            "abs_error_per_feature": dict(zip(FEATURE_NAMES, np.abs(w_hat - w_true).round(6).tolist()))}


def heldout_evaluation(test_mdps, test_demos, w_true, policies: dict) -> dict:
    """``policies[name]`` is a callable mdp -> policy array. Returns metrics averaged over documents."""
    res = {}
    for name, make in policies.items():
        nll, tv, agree, ret, fe, fe_emp = [], [], [], [], [], []
        for mdp, (S_, A_) in zip(test_mdps, test_demos):
            pi_true, _ = soft_value_iteration(mdp, w_true)
            pi = make(mdp)
            nll.append(-mean_log_likelihood(pi, S_, A_))
            c = policy_comparison(mdp, pi_true, pi)
            tv.append(c["tv_distance"])
            agree.append(c["argmax_agreement"])
            ret.append(expected_true_return(mdp, pi, w_true))
            fe.append(feature_expectations(mdp, np.broadcast_to(pi, (mdp.horizon,) + pi.shape[-2:])))
            fe_emp.append(empirical_features(mdp, S_, A_))
        res[name] = {"heldout_nll_per_step": float(np.mean(nll)), "tv_distance_to_true_policy": float(np.mean(tv)),
                     "argmax_agreement_with_true_policy": float(np.mean(agree)),
                     "expected_true_return": float(np.mean(ret)),
                     "feature_expectations": dict(zip(FEATURE_NAMES, np.mean(fe, 0).round(4).tolist()))}
        res["_empirical_heldout_demos"] = {"feature_expectations": dict(zip(FEATURE_NAMES, np.mean(fe_emp, 0).round(4).tolist()))}
    return res


def run_irl_experiment(cfg: dict, make_figures: bool = True, log=print) -> dict:
    """Reward-recovery experiment: writes results/metrics/irl_results.json and IRL figures."""
    t0 = time.perf_counter()
    ic = cfg["irl"]
    seed = cfg["seed"]
    w_true = np.asarray(ic["true_weights"], dtype=np.float64)
    off = ic["doc_id_offset"]
    train_ids = list(range(off, off + ic["n_train_docs"]))
    test_ids = list(range(off + ic["n_train_docs"], off + ic["n_train_docs"] + ic["n_test_docs"]))
    train_mdps, _ = make_mdps(cfg, train_ids)
    test_mdps, test_docs = make_mdps(cfg, test_ids)
    train_demos = sample_demos(train_mdps, w_true, ic["trajectories_per_doc"], seed, 1)
    test_demos = sample_demos(test_mdps, w_true, ic["trajectories_per_doc"], seed, 2)

    w_irl, hist = maxent_irl(train_mdps, train_demos, ic["n_iters"], ic["lr"], ic["l2"])
    th_bc = fit_logit_bc(train_mdps, train_demos, ic["n_iters"], ic["lr"])
    ll_true, _ = irl_objective_and_grad(train_mdps, train_demos, w_true)
    log(f"[irl] w_true={w_true.round(3).tolist()} w_irl={w_irl.round(3).tolist()} "
        f"iters={len(hist)} final grad norm={hist[-1]['grad_norm']:.2e}")

    policies = {
        "true_reward_soft_optimal": lambda m: soft_value_iteration(m, w_true)[0],
        "irl_soft_optimal": lambda m: soft_value_iteration(m, w_irl)[0],
        "bc_logit_one_step": lambda m: logit_policy(m, th_bc),
        "uniform_valid_actions": uniform_policy,
    }
    held = heldout_evaluation(test_mdps, test_demos, w_true, policies)

    sweep = []
    for n in ic["sample_sizes"]:
        # nested subsets: the first n of the 20 trajectories per page (n = 20 is the main run)
        demos_n = [(S_[:n], A_[:n]) for S_, A_ in train_demos]
        w_n, h_n = maxent_irl(train_mdps, demos_n, ic["n_iters"], ic["lr"], ic["l2"])
        nll = np.mean([-mean_log_likelihood(soft_value_iteration(m, w_n)[0], S_, A_)
                       for m, (S_, A_) in zip(test_mdps, test_demos)])
        sweep.append({"trajectories_per_doc": n, "n_train_trajectories": n * len(train_mdps), "w": w_n.tolist(),
                      **recovery_metrics(w_n, w_true), "heldout_nll_per_step": float(nll), "iters": len(h_n)})
        log(f"[irl] sweep n={n}/doc: cos={sweep[-1]['cosine_similarity']:.4f} "
            f"rel.err={sweep[-1]['relative_l2_error']:.3f} heldout NLL={nll:.4f}")

    rec = recovery_metrics(w_irl, w_true)
    results = {
        "note": "Controlled reward-recovery test: demonstrations come from a soft-optimal agent with a KNOWN linear "
                "reward on a discretized grid of synthetic documents. Not human data; not the rule-based P0 expert.",
        "feature_names": list(FEATURE_NAMES), "action_names": list(ACTION_NAMES),
        "grid": {"rows": ic["grid_rows"], "cols": ic["grid_cols"], "horizon": ic["horizon"]},
        "train_doc_ids": train_ids, "test_doc_ids": test_ids, "trajectories_per_doc": ic["trajectories_per_doc"],
        "w_true": w_true.tolist(), "w_irl": w_irl.tolist(), "theta_bc_logit": th_bc.tolist(),
        "recovery": rec,
        "optimization": {"iterations": len(hist), "final_grad_norm": hist[-1]["grad_norm"],
                         "final_train_log_likelihood_per_traj": hist[-1]["log_likelihood"],
                         "train_log_likelihood_per_traj_at_w_true": ll_true,
                         "initial_train_log_likelihood_per_traj": hist[0]["log_likelihood"]},
        "heldout": held, "sample_size_sweep": sweep,
        "success_criterion": "cosine >= 0.99 and relative L2 error <= 0.15 with the full training set",
        "reward_recovered": bool(rec["cosine_similarity"] >= 0.99 and rec["relative_l2_error"] <= 0.15),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    mdir = results_subdir(cfg, "metrics")
    with open(mdir / "irl_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(results_subdir(cfg, "logs") / "irl_optimization.csv", "w", encoding="utf-8") as f:
        f.write("iter,log_likelihood,grad_norm," + ",".join(f"w_{n}" for n in FEATURE_NAMES) + "\n")
        for h in hist:
            f.write(f"{h['iter']},{h['log_likelihood']:.8f},{h['grad_norm']:.8e}," + ",".join(f"{x:.6f}" for x in h["w"]) + "\n")
    save_config(cfg, results_subdir(cfg, "logs") / "irl_config.yaml")
    if make_figures:
        _irl_figures(cfg, results, hist, test_mdps[0], test_docs[0], w_true, w_irl, th_bc)
    log(f"[irl] cosine={rec['cosine_similarity']:.4f} rel.err={rec['relative_l2_error']:.3f} "
        f"held-out NLL: true={held['true_reward_soft_optimal']['heldout_nll_per_step']:.4f} "
        f"irl={held['irl_soft_optimal']['heldout_nll_per_step']:.4f} "
        f"bc={held['bc_logit_one_step']['heldout_nll_per_step']:.4f} "
        f"uniform={held['uniform_valid_actions']['heldout_nll_per_step']:.4f}")
    return results


def _irl_figures(cfg, results, hist, mdp, doc, w_true, w_irl, th_bc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = results_subdir(cfg, "figures")
    x = np.arange(len(FEATURE_NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(x - 0.27, w_true, 0.27, label="w_true (ground truth)", color="#1f77b4")
    axes[0].bar(x, w_irl, 0.27, label="w recovered by MaxEnt IRL", color="#d62728")
    axes[0].bar(x + 0.27, th_bc, 0.27, label="one-step BC logit weights (not a reward)", color="#999999")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FEATURE_NAMES, fontsize=8)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    rec = results["recovery"]
    axes[0].set_title(f"Reward weights (cosine {rec['cosine_similarity']:.3f}, rel. L2 error "
                      f"{rec['relative_l2_error']:.3f})", fontsize=10)
    ll = [h["log_likelihood"] for h in hist]
    axes[1].plot(ll, color="#d62728", label="IRL estimate")
    axes[1].axhline(results["optimization"]["train_log_likelihood_per_traj_at_w_true"], color="#1f77b4", ls="--",
                    label="value at w_true")
    axes[1].set_xlabel("Adam iteration")
    axes[1].set_ylabel("train log-likelihood per trajectory")
    axes[1].legend(fontsize=8)
    axes[1].set_title("MaxEnt IRL optimisation", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "irl_weights.png", dpi=110)
    plt.close(fig)

    R, C, m = mdp.rows, mdp.cols, cfg["document"]["margin"]
    H, W = doc.image.shape
    extent = (m, W - m, H - m, m)
    pols = [("expert: soft-optimal for w_true", soft_value_iteration(mdp, w_true)[0]),
            ("MaxEnt IRL policy (from learned w)", soft_value_iteration(mdp, w_irl)[0]),
            ("one-step BC logit policy", np.broadcast_to(logit_policy(mdp, th_bc), (mdp.horizon, R * C, 4)))]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    axes[0].imshow(doc.image, cmap="gray", vmin=0, vmax=255)
    dim = axes[0].imshow(mdp.density.reshape(R, C), cmap="viridis", alpha=0.45, extent=extent, vmin=0, vmax=1)
    fig.colorbar(dim, ax=axes[0], fraction=0.046, label="landing density feature")
    axes[0].set_title(f"held-out document {doc.doc_id}: grid cells", fontsize=9)
    visits = [state_visitation(mdp, pi).sum(0).reshape(R, C) for _, pi in pols]
    vmax = max(float(v.max()) for v in visits)  # one colour scale for all three policies
    for ax, (name, _), D in zip(axes[1:], pols, visits):
        ax.imshow(doc.image, cmap="gray", vmin=0, vmax=255)
        im = ax.imshow(D, cmap="magma", alpha=0.6, extent=extent, vmin=0, vmax=vmax)
        ax.set_title(f"{name}\nexpected visits per cell over {mdp.horizon} steps (max {D.max():.1f})", fontsize=9)
    fig.colorbar(im, ax=axes[1:], fraction=0.015, label="expected visits (shared scale)")
    for ax in axes:
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_axis_off()
    fig.savefig(fig_dir / "irl_visitation.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    import argparse

    from .config import load_config

    ap = argparse.ArgumentParser(description="Run the MaxEnt IRL reward-recovery experiment")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    run_irl_experiment(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
