# Methodology

This file documents exactly what the code does. Every rule below can be found in the module named
in its heading. Everything is synthetic: no human eye-tracking data is used anywhere.

## 1. Documents (`document_generator.py`)

* Grayscale page, 600 x 800 px, white background, 40 px margins.
* One heading (font 26-30 px, drawn with a 1 px stroke so it looks bold, 2-5 words) and 3-4
  paragraphs of 2-5 lines each.
* Each paragraph draws one style: **dense** (font 13-14 px, line spacing 1.3, word gap 4-6 px,
  full width), **normal** (16-17 px, 1.6, 6-9 px) or **sparse** (19-21 px, 2.0, 10-16 px, 55-75 %
  width). The last line of a paragraph is 30-90 % of the paragraph width.
* Text gray level: high contrast (0-40) or, with probability 0.3, low contrast (120-160).
  With probability 0.25 a paragraph sits on a light gray box (222-238).
  `word.contrast = (background gray - text gray) / 255`.
* Words are random letter strings (2-11 letters, English letter frequencies). They have no meaning,
  so nothing here models language processing.
* Font: Pillow's bundled default FreeType font (no system fonts, same on every machine).
* Document `i` is generated from `np.random.default_rng([seed, i])`; layout (word and line boxes,
  in reading order) is saved next to the PNG as JSON.

## 2. Feature maps and saliency (`saliency.py`)

All maps are float32, same size as the page, values in [0, 1].

| map | definition |
|---|---|
| `ink` | adaptive threshold: pixel is ≥ 25 gray levels darker than its 25 x 25 local mean |
| `density` | 15 x 15 box mean of `ink` (fraction of ink pixels) |
| `text_mask` | morphological closing of `ink` with a 15 x 5 kernel, then 3 x 3 dilation |
| `contrast` | local std. dev. of intensity (Gaussian σ = 3 px) / 127.5 |
| `edges` | Sobel gradient magnitude / (4·255), Gaussian σ = 2 px |
| `saliency` | `minmax( G_σ=4 * (0.5·contrast + 0.5·edges) )` |

`minmax` returns all zeros when the map is constant, so blank pages give zeros, never NaNs.
Saliency is a handcrafted proxy for **visual prominence**. It is not a learned or validated model
of human gaze. Density and saliency are kept as separate features. Low contrast halves saliency
(0.31 vs 0.61 mean word saliency in dense paragraphs, documents 0-29) but changes ink density only a
little (0.179 vs 0.214). "Density" is the local fraction of dark pixels. It tracks font size and
stroke weight (bold heading 0.425, sparse 0.250, dense 0.214), not how much text a paragraph contains.

## 3. Environment, state and action (`environment.py`)

* **Coordinates:** gaze `g = (x, y)` in [0, 1]², `x = column / 600`, `y = row / 800`, origin top-left.
  Pixels are only used for drawing and for reporting errors in px.
* **Observation at time t** (a function of the page and `g_t` only):
  * `saliency`: tensor `1 x 84 x 84`, values in [0, 1]. It is a window of the saliency map centred
    on `g_t` covering 0.5 x 0.25 of the page (300 x 200 px), resampled to 84 x 84 (so the scale is
    3.57 px per cell horizontally and 2.38 px per cell vertically), with zeros outside the page.
  * `features`: 7 numbers `[x, y, saliency, density, contrast, edges, text_occupancy]`; the last five
    are means of the maps over a 40 x 16 px window around `g_t`. All are in [0, 1]. They are
    z-scored with training-set mean and standard deviation before entering the network.
* **Action:** displacement `(dx, dy)` in normalised units, and the dwell time of the current
  fixation in synthetic milliseconds.
* **Transition:** `g_{t+1} = clip(g_t + (dx, dy), 0, 1)`. Non-finite displacements or non-positive
  dwells raise an error.
* **Termination:** after `max_steps` steps (for rollouts, the length of the matching expert
  trajectory).

## 4. Synthetic expert (`expert.py`)

The expert knows the layout (word boxes in reading order). At a fixation on word `i`:

1. Stop if `i` is the last word (or 300 fixations have been made).
2. Regression with probability `0.04 + 0.10·(1 − contrast_i)`: jump back `k ~ U{1,2,3}` words,
   never before the first word of the previous line. Reading continues forward from there.
3. Else, if `i` is the last word of its line: return sweep to the first word of the next line.
4. Else go to word `i+1`, or skip it (land on `i+2` on the same line) with probability
   `p·(1 − 0.5·saliency_{i+1})`, `p = 0.45` for words of ≤ 3 letters and `0.04` otherwise.

Landing point: `x = x0 + clip(N(0.40, 0.12), 0, 1)·width`, `y = line centre + N(0, 0.15·line height)`.
This is a simulation. It is not a fitted model of human oculomotor behaviour.

**Dwell rule** (synthetic ms) for a fixation on word `i`:

```
clean_i = 140 + 18·n_chars_i + 120·density_i + 100·(1 − contrast_i)
dwell_i = max(80, clean_i · exp(ε)),   ε ~ N(0, 0.10²)
```

`density_i` is the mean of the `density` map inside the word box. The noise level 0.10 was fixed
before any model was trained. Because `ε` cannot be observed, the correlation between `clean` and
`dwell` on held-out data (the **noise ceiling**, about 0.81) is an upper bound for any predictor.
Most of the explainable variance comes from word length (on validation documents, word length alone
correlates r = 0.69 with dwell).

## 5. Dataset and splits (`dataset.py`)

100 documents × 5 trajectories. A transition `t → t+1` stores the current gaze, the stored scalar
features, the next gaze, the displacement target, the dwell of fixation `t`, plus `dwell_clean`, the
saccade type and the word index for analysis only (never model inputs). The last fixation of each
trajectory has no next fixation and is not a training row. The saliency crop is recomputed from the
saved image when needed (it is a deterministic function of image and gaze).

Documents are split 70 / 15 / 15 (train / val / test) by **document id** with a seeded permutation.
The split is stored in `data/processed/splits.json` and checked to be disjoint every time it is
loaded. Every trajectory therefore belongs to exactly one split.

## 6. Behavior Cloning model (`model.py`, `train.py`)

```
saliency crop 1x84x84 -> Conv(16,5x5,s2) -> ReLU -> Conv(32,3x3,s2) -> ReLU -> Conv(32,3x3,s2) -> ReLU
                      -> flatten (32x11x11) -> Linear(128) -> ReLU                     ┐
features (7)          -> Linear(64) -> ReLU                                            ┤ concat
                                                                                       ┘
  -> Linear(128) -> ReLU -> Dropout(0.1) -> Linear(128) -> ReLU -> { movement head: Linear(2)
                                                                   { dwell head:    Linear(1)
```

The shuffled-crop control (`--variant shuffled_saliency`) is exactly the full model and training.
In its datasets, transition `i` is paired with the saliency crop of transition `perm[i]` of the same
split, where `perm` is a fixed random permutation (`train.make_datasets(shuffle_crops=True)`). The
input statistics are those of real crops, but the crop is unrelated to the current gaze. It is
evaluated teacher-forced only (a free-running version is not meaningful). An earlier all-zero-input
control was removed: a constant input only adds a bias to the trunk, so it is equivalent to the
feature-only model by construction.

The feature-only ablation (Baseline B) builds the model with `use_saliency=False`. The CNN branch is
then not built at all (there are no conv weights and no saliency crop), and the trunk sees only the
64-d feature embedding.

**Targets.** Movement: `z_m = (d − μ_d) / σ_d` per axis. Dwell: `z_w = (log w − μ_log) / σ_log`.
All statistics come from the training split. Inverse: `d = z_m·σ_d + μ_d`, `w = exp(z_w·σ_log + μ_log)`,
so predicted dwell is always positive. Because both targets are z-scored, neither loss term
dominates just because of units.

**Loss.** `L = λ_m · MSE(ẑ_m, z_m) + λ_w · MSE(ẑ_w, z_w)` with `λ_m = λ_w = 1`. MSE regression
predicts the conditional mean. Where the expert's next move is random (skip vs. no skip,
regression vs. forward), the best MSE prediction is an average of the possible moves, so part of
the movement error cannot be removed.

**Training.** AdamW (lr 1e-3, weight decay 1e-4), batch 256, at most 30 epochs, early stopping
with patience 5 on the validation loss. The checkpoint with the best validation loss is kept.
Python, NumPy and PyTorch are seeded. The test split is never loaded during training.

## 7. Evaluation (`evaluate.py`)

* **Teacher-forced, one step ahead** (main metrics): every held-out expert transition is fed with
  the expert's current fixation. Dwell: Pearson r, MAE and RMSE in synthetic ms, R². Movement:
  Euclidean error of the predicted next fixation, in normalised units and in px
  (`sqrt((Δx·600)² + (Δy·800)²)`), overall and per saccade type, plus the fraction of predicted
  next fixations that land on the text mask. Pearson r is reported as `null`/"undefined" when an
  input is constant.
* **Baseline A (constant):** training-set mean dwell and mean displacement for every test example.
* **Baseline B / ablation (features only):** same training procedure, no saliency map/crop (the scalar
  window-mean saliency feature stays). **Shuffled-crop control:** see section 6.
* **Noise-ceiling reference:** the expert's own noise-free dwell (`dwell_clean`) scored like a
  model. It uses hidden variables and is only an upper bound, not a competitor.
* **Free-running rollout:** for each test document, start at the first fixation of expert
  trajectory 0 and let a policy move with its *own* outputs for the same number of steps. Scored with
  layout information (used only for scoring): fraction of fixations on the text mask, fraction of
  words fixated (word coverage), fraction of text lines visited, fraction of steps clipped at the
  page border, mean predicted dwell, and the number of distinct positions.
* All test predictions are written to `results/metrics/predictions_test_seed<k>.csv`. The test
  `metrics_from_predictions_csv` recomputes the reported numbers from that file.

## 8. Development history (validation set only, raw logs in `development_log.md`)

The test split was not evaluated until the configuration was final. Four changes were tried and
judged **on validation documents only** (full model, seed 0):

| step | change | val dwell r | val move error (px) | kept? |
|---|---|---|---|---|
| v0 | 60 docs × 6 traj., CNN pooled to 4 x 4 | 0.646 | 33.3 | – |
| A | no pooling (flatten 11 x 11) | 0.650 | 31.6 | yes |
| B | A + saliency blur σ 4 → 1.5 | 0.649 | 30.7 | no (no clear gain) |
| C | A + 100 docs × 5 traj. | 0.701 | 30.2 | **yes (final)** |
| D | A + 200 docs × 3 traj. | 0.707 | 28.2 | no (outside the spec's suggested range, small gain) |

The dwell noise level, the expert rules and the metric definitions were not changed at any point.


## 9. MaxEnt IRL extension (`irl.py`)

A separate, controlled experiment. It does not use the BC network or the rule-based P0 expert.
The rule-based expert has no explicit reward function, so reward recovery cannot be scored on it.
Instead, the demonstrations come from an agent that is soft-optimal for a **known** linear reward.

**MDP per document.** The page area inside the margins is cut into an 8 x 6 grid (cells of
about 87 x 90 px); `s = row·6 + col`. Each episode starts in cell (0, 0) and lasts T = 30 steps. Actions:
`right` (col+1), `skip` (col+2), `regress` (col−1), `sweep` (first cell of the next row). Moves that
leave the grid are masked out, so the dynamics are deterministic and every state keeps at least one
legal action. Documents use ids 1000-1029 (train) and 1030-1044 (test), generated with the same P0
document generator. They are disjoint from the BC documents.

**Features and reward.**
`φ(s,a) = [saliency(s'), density(s'), 1[regress], 1[skip], 1[sweep]]`. `s'` is the landing
cell. The cell saliency and density are area averages of the P0 maps, min-max scaled to [0, 1] per
document. `right` is the reference action. `R(s,a) = wᵀφ(s,a)`, with ground truth
`w_true = [1.0, 2.0, −1.5, −0.5, −1.0]` (fixed in the config before the first IRL run).

**Expert.** Finite-horizon soft value iteration:

```
Q_t(s,a) = R(s,a) + V_{t+1}(s'),   V_t(s) = log Σ_a exp Q_t(s,a),   V_T = 0,   π_t(a|s) = exp(Q_t − V_t)
```

20 trajectories are sampled per document from `π_t`.

**MaxEnt IRL.** With deterministic dynamics, `log P(τ|w) = wᵀΦ(τ) − V_0(s_0)`. The average
log-likelihood is concave in `w` and its gradient is

```
∇_w = mean_τ Φ(τ) − E_{π_w}[Φ]       (expert feature counts − learner feature expectations)
```

`E_{π_w}[Φ]` is computed exactly with a forward pass over state-visitation frequencies
`D_{t+1}(s') = Σ_{s,a} D_t(s) π_t(a|s) 1[s' = next(s,a)]`. The gradient is averaged over training
documents and `w` is updated by Adam gradient ascent (lr 0.05, 500 iterations, starting at w = 0). The
derived policy is the soft-optimal policy for the learned `w` (soft value iteration again).

**Comparisons on held-out documents:** the true policy (an upper reference), the IRL policy, a
**one-step BC** conditional logit `π(a|s) ∝ exp(θᵀφ(s,a))` fitted by maximum likelihood on the same
demonstrations and features (no planning), and a uniform policy over legal actions.
Metrics: held-out negative log-likelihood per step, visitation-weighted total-variation distance to
the true policy, visitation-weighted argmax agreement, expected ground-truth return, and feature
expectations.

**Reward recovery** is reported as cosine similarity and relative L2 error between `w` and `w_true`.
Success was defined before running as cosine ≥ 0.99 and relative error ≤ 0.15.
Identifiability: adding a constant to every reward does not change the policy (all trajectories have
length T). There is no constant feature, and `right` is the reference action, so this does not
arise here. The scale *is* identified, because the expert's temperature is fixed to 1. Potential-based
shaping (Ng, Harada & Russell, 1999) would also leave the optimal behaviour unchanged, so in general
only an equivalence class of rewards can be recovered. The recovery here works because the
generating reward is inside the linear model class.

**Known limitation of the grid expert.** The reward has no memory of what has been read, so the
soft-optimal agent goes back and forth over dense cells: about 11 regressions per 30 steps.
Changing the regress weight to −2.5, −3 or −4 was checked on expert behaviour alone; it still gave
9.5, 8.3 and 6.2 regressions, so this is structural. `w_true` was left at its pre-specified value. This expert is a
reward-recovery testbed, not a realistic reader.

**Tests** (`tests/test_irl.py`): shapes and legal-action masks; probabilities sum to 1; `V_0` equals
a brute-force log-sum-exp over all action sequences; exact feature expectations match Monte-Carlo
estimates; the analytic gradient matches finite differences; recovery of a known reward on small
random MDPs; the experiment writes its outputs.
