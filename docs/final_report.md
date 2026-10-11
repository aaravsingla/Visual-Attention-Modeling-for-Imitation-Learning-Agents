# Final report

Project: *Visual Attention Modeling for Imitation Learning: A Lightweight Simulation of Reading Behavior*
(`visual-attention-imitation/`). Everything below was run on the development machine: Windows 11,
Python 3.11.9, PyTorch 2.14.1 on CPU, no GPU, no API keys, no external data.

**How to read this report.** "Measured" means I ran it and read the output files.
"Expected" or "not shown" means it was not measured. No number here was typed in by hand. Each one
comes from `results/metrics/*.json|csv` or `results/logs/*` written by the commands listed in section 2.

---

## 1. What was built

| module | what it does |
|---|---|
| `document_generator.py` | Synthetic 600 x 800 px pages: a heading plus 3-4 paragraphs of pseudo-words in dense / normal / sparse styles, high or low contrast, some on gray boxes. Seeded per document id. Layout saved as JSON. |
| `saliency.py` | Handcrafted, deterministic maps: ink, ink density, text mask, local contrast, edges, and saliency = blurred contrast + edges, min-max scaled to [0, 1]. Blank images give zeros, never NaN. |
| `environment.py` | Gaze in normalised [0, 1]² page coordinates. Observation = 1x84x84 saliency crop centred on the gaze + 7 scalar features. Action = (dx, dy) + dwell. Clipping, termination, rendering. |
| `expert.py` | Transparent rule-based reader: next word, skip of short words, contrast-dependent regressions, return sweeps, landing noise. Explicit dwell rule (word length, ink density, contrast, log-normal noise σ = 0.10) in synthetic ms. |
| `dataset.py` | 100 documents x 5 trajectories = 59,616 transitions in `transitions.npz`. Document-level 70/15/15 split, checked for disjointness on every load. Train-only normaliser. PyTorch `Dataset`. |
| `model.py`, `train.py` | Two-headed BC policy (CNN on the crop + MLP on the features → shared trunk → movement head and dwell head). z-scored targets (log-dwell). AdamW, early stopping, best-validation checkpoint, CSV logs, CLI. Three variants: `full`, `no_saliency` (no CNN, no crop), `shuffled_saliency` (control: same network, crops from other random transitions). |
| `evaluate.py` | Held-out teacher-forced metrics: dwell r / MAE / RMSE / R², next-fixation error in normalised units and px, per-saccade-type errors. Constant baseline from training means. Noise-ceiling reference. Free-running rollouts. Prediction CSV, figures. |
| `visualize.py` | Expert / learned / comparison scanpaths (first 40 fixations), full-length rollouts, saliency overlay, loss curves, dwell scatter, ablation bars. |
| `irl.py` | Separate 8 x 6 grid MDP per page, with a soft-optimal expert under a **known** linear reward. Soft value iteration, exact visitation-frequency feature expectations, MaxEnt IRL (Adam ascent on the exact gradient), policy derived from the learned reward, one-step BC logit and uniform baselines, held-out evaluation, data-size sweep, figures, CLI. |
| `scripts/` | `generate_data.py`, `run_experiments.py` (BC variants, multi-seed, IRL), `reproduce_results.py` (one command), `dev_validation_run.py` (validation-only development runs). |
| `tests/` | 56 pytest tests (data, saliency, environment, expert, dataset, model, training, metrics, rollouts, IRL). |
| `docs/` | `methodology.md`, `experiments.md`, `interview_notes.md`, `development_log.md`, `figures/`, this report. |

Not built (by design or by scope): web apps, APIs, Docker, transformers, LLMs, hyperparameter
searches, pretrained models, human data, IRL on the rule-based expert, and P2 items other than
multiple seeds.

## 2. What works (commands executed and checked)

| command | verified outcome |
|---|---|
| `py -3.11 -m venv .venv` + `pip install torch==2.14.1 --index-url .../cpu` + `pip install -r requirements.txt` + `pip install -e .` | Installed; `import torch, cv2, numpy, matplotlib, sklearn, yaml, PIL` works. |
| `python -m pytest -q` | **56 passed, 0 failed.** |
| `python scripts/reproduce_results.py --clean` | Exit 0 in **308.1 s**. Deletes `data/`, `checkpoints/`, `results/` and regenerates all seed-0 artifacts and IRL. Seed-0 metrics identical to the full run. |
| `python scripts/reproduce_results.py --clean --seeds 0 1 2 --copy-figures` | Exit 0 in **1039.4 s**. Produces every artifact, including the three-seed summary, and copies the README figures into `docs/figures/`. |
| Every command in the README "Individual stages" block and the two override examples | All 14 commands (installs, tests, `generate_data`, the three `train` variants, `evaluate`, `evaluate --splits val`, `run_experiments --skip-train --skip-irl`, `python -m visual_attention.irl`, and the two scratch-path examples) ran **exactly as written** with exit code 0. Retraining stage by stage reproduced `test_metrics_seed0.json`, `summary_seed0.csv` and `predictions_test_seed0.csv` byte for byte. The validation-only run wrote only `*_val_only` files. (`run_experiments --seeds 0 1 2 --skip-irl` was run in earlier rounds; in the final round the same code path ran inside `reproduce_results.py --seeds 0 1 2`.) |
| Reproducibility | Repeated clean runs gave byte-identical `predictions_test_seed0.csv`, `summary_seed0.csv`, `multiseed_summary.json`, `transitions.npz` and `splits.json`. The JSON metric files differed only in wall-clock seconds and absolute paths. Both independent reviewers reproduced the numbers. |

## 3. Test results

**56 passed, 0 failed, 0 skipped** (`python -m pytest -q`, about 30-50 s). No test was weakened. One
assertion was changed during development, and the reason is recorded: the text-mask coverage test was
moved to the central band of each word. Tight word boxes include sparse ascender/descender rows, so
the original whole-box check did not test the mask properly. The test also gained a new check that
the mask does not fire away from the text.

## 4. Measured results (held-out test set: 15 documents, 8,857 transitions; seed 0)

Teacher-forced, one step ahead. Dwell in synthetic ms:

| model | dwell Pearson r | dwell MAE | dwell RMSE | dwell R² | next-fixation error |
|---|---|---|---|---|---|
| constant baseline (train means) | undefined | 38.7 | 48.6 | −0.006 | 88.1 px (0.146) |
| BC features only (saliency ablation) | 0.585 | 30.8 | 39.5 | 0.334 | 56.3 px (0.093) |
| BC shuffled-crop control | 0.583 | 30.9 | 40.0 | 0.319 | 57.0 px (0.094) |
| **BC full (saliency crop + features)** | **0.671** | **28.1** | **36.0** | **0.447** | **31.7 px (0.052)** |
| noise ceiling (expert's noise-free rule, reference only) | 0.807 | 22.5 | 28.7 | 0.650 | – |

Three training seeds (mean ± sample std): full r = **0.673 ± 0.006**, MAE 28.19 ± 0.32 ms,
30.55 ± 0.97 px; features only 0.592 ± 0.006, 30.75 ± 0.09 ms, 56.52 ± 1.84 px; shuffled crops
0.590 ± 0.007, 30.81 ± 0.22 ms, 56.22 ± 1.17 px.

* **The resume's "85 % correlation" is not reproduced.** The measured held-out Pearson r is 0.671
  (0.673 ± 0.006 over seeds). It is below the 0.807 ceiling that the chosen synthetic noise allows.
  The noise level was fixed before any training and never changed. All tuning (4 development runs)
  used the validation split only (`docs/development_log.md`). Likely reasons for the gap to the
  ceiling: the policy must infer word length from a blurred, downsampled saliency crop (the expert
  reads it from exact word boxes), and the full model overfits after about 6 epochs (70 training documents).
* **Saliency ablation:** removing the crop raises next-fixation error from 31.7 to 56.3 px (return
  sweeps 56.0 → 186.8 px) and lowers dwell r from 0.671 to 0.585. The shuffled-crop control (same
  552k-parameter network, real crops from other transitions) is as bad as the 26k-parameter
  feature-only model. So the gain requires the crop at the current gaze; capacity or input
  statistics alone do not give it.
* **Error analysis:** regressions (random in the expert) are the worst case for every model. The full
  model does worse than the constant baseline on them (166.6 vs 139.4 px).
* **Independent reviews:** two fresh reviewer subagents audited the code and reran the pipeline.
  Review 1 found 1 major issue (the documented command crashed on the not-yet-written IRL module) and
  11 minor ones. Review 2 found 4 major issues: the zero-input control was invalid by construction,
  validation-only evaluation could overwrite the test file, the IRL figure used per-panel colour
  scales, and the resume IRL claim was not assessed. It also found 14 minor ones. All were fixed and
  the pipeline rerun.
* **Free-running rollouts:** fixations on text: expert 98.4 %, full 63.7 %, features only 35.9 %,
  constant 30.3 %. Words fixated: 91.9 / 60.8 / 34.5 / 11.4 %. The learned policy reads left to right
  with return sweeps but skips lines, leaves the text and drifts to the bottom-left corner
  (10 of 15 rollouts end exactly at (0, 1); 17.8 % of steps clipped). This is a compounding-error effect
  that teacher-forced metrics do not show.

## 5. IRL status: implemented and tested (controlled reward recovery)

* MaxEnt IRL on a separate 8 x 6 grid MDP (horizon 30). The demonstrations come from a soft-optimal
  expert whose linear reward is **known**: `w_true = [1.0, 2.0, −1.5, −0.5, −1.0]` over [landing
  saliency, landing density, regress, skip, sweep]. 30 training pages and 15 held-out pages, 20
  trajectories each.
* Recovered `w = [0.902, 2.001, −1.463, −0.522, −1.071]`: cosine **0.999**, relative L2 error **0.044**,
  meeting the criterion set before the run (≥ 0.99 / ≤ 0.15; lenient — even 1 trajectory per page meets it).
* Policy derived from the learned reward, on held-out pages: NLL per step 0.8253 (true policy 0.8251;
  one-step BC logit 0.9197; uniform 1.2958), TV distance to the true policy 0.006, argmax agreement
  99.3 %. Its feature expectations match the expert's (e.g. regressions 11.05 vs 10.94 per episode).
* Tests: a brute-force check of `V_0`, a finite-difference gradient check, a Monte-Carlo check of the
  feature expectations, recovery on random small MDPs, and output files.
* Limits: the expert is inside the model class (this validates the method and code, not a theory of
  reading). The comparison with one-step BC is structurally favourable to IRL. The memoryless reward
  makes the grid expert oscillate (about 11 regressions per 30 steps). Cell saliency and density are
  collinear (r = 0.88 on a held-out page). No human data. IRL was **not** run on the rule-based P0
  expert (it has no reward to compare with). The resume's IRL bullet ("processing visual saliency
  maps as state inputs", "mimics human gaze") is **not supported as worded**; see README §13b.

## 6. Runtime (measured, CPU)

* `reproduce_results.py --clean`: 308.1 s (8.2 s data, 262.2 s for training and evaluating the three
  BC variants, 37.8 s IRL). Earlier runs of the previous code version took 339-341 s.
* With `--seeds 0 1 2 --copy-figures`: 1039.4 s.
* Single model training (seed 0): full 116.2 s, features only 6.6 s, shuffled-crop control 128.7 s.

## 7. Generated artifacts (all paths relative to `visual-attention-imitation/`)

* Data: `data/raw/doc_XXXX.png|json`, `data/processed/transitions.npz`, `data/processed/splits.json`,
  `data/processed/config.yaml`.
* Checkpoints: `checkpoints/{full,no_saliency,shuffled_saliency}_seed{0,1,2}/best.pt` (+ `config.yaml`,
  `train_summary.json`).
* Metrics: `results/metrics/test_metrics_seed{0,1,2}.json`, `summary_seed{0,1,2}.csv`,
  `predictions_test_seed{0,1,2}.csv`, `multiseed_summary.{json,csv}`, `irl_results.json`,
  `eval_config_seed*.yaml`.
* Logs: `results/logs/train_<variant>_seed<k>.csv`, `irl_optimization.csv`, `reproduce_timing.json`,
  `run_experiments_timing_*.json`, `run_config.yaml`, `irl_config.yaml`.
* Figures: `results/figures/` — `saliency_example.png`, `expert_scanpath_example.png`,
  `loss_curves.png`, `dwell_scatter_{full,no_saliency,shuffled_saliency}.png`, `ablation.png`,
  `ablation_multiseed.png`, `scanpath_doc{0,10}_{expert,learned,comparison,full_rollout}.png`,
  `irl_weights.png`, `irl_visitation.png`. Copies of the README figures are in `docs/figures/`.

## 8. Known limitations

* **No human eye-tracking data anywhere.** All agreement is with a synthetic expert written in this
  repository. Nothing shows human-like reading.
* The dwell correlation is set largely by the generator (the noise level, and how visible the dwell
  factors are). Only comparisons within this generator mean something.
* Pseudo-words: no linguistic effects (frequency, predictability).
* "Ink density" tracks font size and stroke weight, not the amount of text.
* BC with MSE predicts the conditional mean, so it cannot reproduce random skips and regressions.
  Rollouts show covariate shift. No DAgger-style fix is implemented.
* 15 test documents from the same generator (in-distribution only); 3 training seeds.
* The IRL experiment is a controlled toy (see section 5).
* Saliency also highlights the borders of gray boxes (a handcrafted-feature artifact).

## 9. Next steps (highest value only)

1. Train and evaluate the same pipeline on a real reading eye-tracking corpus with participant- and
   text-disjoint splits, with an inter-reader noise ceiling.
2. Fix rollout drift: DAgger-style data aggregation (query the expert on states the policy visits) or
   a stochastic (mixture) action head instead of MSE regression.
3. Make the dwell head see word extent directly, e.g. a second, higher-resolution crop channel, and
   test whether the gap to the noise ceiling closes (on validation first).

## 10. Interview readiness: five details to know before discussing the project

1. **The data are synthetic and you wrote the expert.** Know its decision rule and the dwell formula
   `140 + 18·letters + 120·density + 100·(1 − contrast)`, times log-normal noise (σ = 0.10). Know why
   r ≤ 0.807 (the noise ceiling), and why "85 % correlation" is not a valid claim here (measured
   0.671; and r is not a percentage).
2. **Inputs and leakage.** The observation is a 300 x 200 px saliency window around the *current* gaze,
   resampled to 84 x 84, plus 7 local features. The split is by document. The normaliser and the
   constant baseline use training data only. The test split was untouched during development.
3. **The loss.** Movement and log-dwell targets are both z-scored with training statistics, and
   `L = MSE_move + MSE_dwell`. Dwell = exp(·), so it is always positive. MSE predicts the conditional
   mean, which explains the poor regression predictions.
4. **What the ablation and the control show.** No crop: 31.7 → 56.3 px, r 0.671 → 0.585. Shuffled
   crops in the same network: no better than no crop. So the crop at the current gaze matters. Also
   know why the first, all-zero control was invalid (a constant input only adds a bias).
5. **BC vs. IRL.** BC regresses actions; it is the neural model. MaxEnt IRL is a separate grid
   experiment: gradient = expert feature counts − expected feature counts under the soft-optimal
   policy, found by soft value iteration and a forward visitation pass. It recovers a known reward
   (cosine 0.999). That validates the code, not a theory of human reading. Teacher-forced metrics
   differ from rollouts (64 % of rollout fixations on text vs. 98 % for the expert).

## 11. Choices I made where the spec left room

* **Observation:** a saliency window centred on the gaze (300 x 200 px resampled to 84 x 84), not the
  whole-page map, so a small CNN can see the local layout. 7 scalar features.
* **Dwell noise:** log-normal, σ = 0.10. Fixed before any training and never changed, even though it
  caps r at about 0.81.
* **Dataset size:** 100 documents × 5 trajectories, inside the spec's suggested range, chosen on the
  validation split. The CNN has no pooling (also chosen on validation).
* **Movement:** continuous (dx, dy) regression with MSE, not a discrete action classifier.
* **Rollout length:** each rollout runs as many steps as the paired expert trajectory.
* **Capacity control:** a shuffled-crop control. It replaced an invalid all-zero-input control.
* **IRL:**
  * a separate 8 x 6 grid MDP with 4 actions and 5 features, horizon 30, and pages with ids 1000+;
  * a soft-optimal expert with a known `w_true`, as you specified;
  * the optional IRL on the rule-based expert's demonstrations was not done.
* **P2:** only the training seed varies (data and splits fixed). The config separates `seed` and
  `data_seed`.
* **Font:** Pillow's bundled font, so pages render the same on any machine.
* **Environment:**
  * The TodoWrite tool was not available, so the plan was a checklist in chat.
  * pip downloaded the packages once from PyPI/pytorch.org. Nothing else used the internet.

---

## Requirement table (reviewer checklist and Section 17)

| Requirement | Evidence in Code | Verification Performed | Status | Remaining Issue |
|---|---|---|---|---|
| 1. Model genuinely trains with PyTorch gradients and optimizer updates | `train.py` `run_epoch`: `zero_grad`, `loss.backward()`, `optimizer.step()` (AdamW); `model.train()`/`eval()`; `torch.set_grad_enabled(False)` for validation | Loss curves fall (full: train 1.14 → 0.53); tests `test_loss_finite_and_backward`, `test_optimizer_step_reduces_loss`, `test_small_training_run`; reviewer 2 checked that trained weights moved far from the initialisation | PASS | – |
| 2. Dwell r from held-out predictions vs. expert targets | `evaluate.py` `predict_dataset` on the test split → `dwell_metrics(true, pred)`; `pearson()` returns undefined for constant inputs | Recomputed from `predictions_test_seed0.csv` (unit test + both reviewers): exact match | PASS | – |
| 3. No test labels or future expert actions in the inputs | Features and crop computed only from the image + current gaze (`environment.local_features`, `extract_crop`); `next_gaze`, `dwell_clean`, `event` never fed to the model | `test_features_are_current_state_only` (every row of a document, exact equality); reviewers recomputed inputs | PASS | Rollout step budget equals the expert's length (evaluation only, documented) |
| 4. Train / val / test documents disjoint | `dataset.split_documents`, `assert_disjoint` on every load; `splits.json` | Tests; 70/15/15 with empty intersections; identical after regeneration | PASS | – |
| 5. Ablation really removes the saliency input | `GazePolicy(use_saliency=False)` builds no CNN and the dataset yields no crop; shuffled-crop control keeps the network but breaks the gaze-crop link | `test_ablation_has_no_saliency_branch` (no conv weights, output independent of a crop); `test_shuffled_crop_control` | PASS | The scalar window-mean saliency feature remains (documented: "no saliency map/crop") |
| 6. Rollouts use the model's own movements | `evaluate.rollout`: `env.step(*policy.act(obs))` from the expert's first fixation only | `test_rollout_uses_policy_outputs_only`; full-length rollout figures show drift away from the expert | PASS | – |
| 7. Baselines fair, no test labels | `constant_baseline(data, train_idx)`; all models share splits, metrics and training procedure | `test_constant_baseline_independent_of_test_labels`; value 284.46 ms = training mean | PASS | Architecture chosen on validation for the full model only (documented) |
| 8. Metrics reproducible from saved artifacts | `predictions_test_seed*.csv`, `metrics_from_predictions_csv`, deterministic reruns | Unit test + reviewers recomputed; clean reruns byte-identical (except timing and paths) | PASS | Rollout metrics need the checkpoints (saved) to recompute |
| 9. IRL genuinely infers a reward and derives a policy | `irl.py`: soft value iteration, exact gradient (expert − learner feature expectations), Adam, policy from `soft_value_iteration(m, w_irl)`; `w_true` only generates demos and scores results | Brute-force, finite-difference and Monte-Carlo tests; recovery cosine 0.999 on held-out pages; reviewer 2 re-derived it and checked extra seeds | PASS | Controlled toy; lenient criterion; structurally favourable comparison with BC (all documented) |
| 10. README, results and interview notes match the code and numbers | README, `docs/*.md` | Both reviewers checked every number against the artifacts; every README command run as written | PASS | Reviewer-2 findings were fixed afterwards. The fixes were checked by me (tests, reruns, rereading), not by a third reviewer. |
| P0: installs and imports | `requirements.txt` (pinned), `pyproject.toml` | Fresh venv, `pip install -e .`, imports | PASS | – |
| P0: documents and saliency generated | `document_generator.py`, `saliency.py` | Figures inspected; blank/constant-image tests | PASS | Gray-box border artifact (documented) |
| P0: valid reading-like expert | `expert.py` | Ordering tests; event mix 74 / 8 / 6 / 12 %; all trajectories reach the last word | PASS | Synthetic, not fitted to humans |
| P0: dataset documented | `dataset.py` docstring, README §4/§9 | Schema matches `transitions.npz` | PASS | – |
| P0: reproducible, document-disjoint splits | `splits.json` | Identical across clean runs | PASS | – |
| P0: model trains and saves a checkpoint | `train.py`, `model.save_checkpoint` | `checkpoints/*/best.pt` produced | PASS | – |
| P0: checkpoint loads and predicts | `model.load_checkpoint`, `Policy` | Evaluation and rollouts from the saved checkpoints; round-trip test | PASS | – |
| P0: dwell r / MAE / RMSE on held-out data | `evaluate.dwell_metrics` | 0.671 / 28.1 / 36.0 ms | PASS | Below 0.85 (reported honestly) |
| P0: movement vs. baseline | `movement_metrics`, constant policy | 31.7 vs 88.1 px | PASS | – |
| P0: expert and learned scanpaths visualised | `visualize.save_scanpath_figures` | PNGs inspected | PASS | – |
| P0: saliency ablation run and recorded | metrics JSON / CSV, `ablation*.png` | Reproduced over 3 seeds | PASS | – |
| P0: test suite passes | `tests/` | 56 passed | PASS | – |
| P0: single documented command | `scripts/reproduce_results.py --clean` | Exit 0, 308.1 s | PASS | – |
| P0: README accurate about what is and isn't implemented | README §3, §8, §13b, §14 | Reviewed twice | PASS | – |
| P0: nothing fabricated | All numbers traced to artifacts | Reviewers traced every number | PASS | – |
| P1: IRL infers reward weights from demonstrations | `irl.maxent_irl` | `w` recovered | PASS | – |
| P1: policy derived from the inferred reward | `soft_value_iteration(m, w_irl)` | Held-out NLL 0.8253, TV 0.006 | PASS | – |
| P1: meaningful reward-recovery / policy evaluation | Known `w_true`, held-out pages, baselines, sweep | Measured and reproduced | PASS | Toy setting, expert in the model class |
| P1: documented separately from BC | README §8 + IRL results, methodology §9, experiments E6 | Read | PASS | – |
| P2: seeds with mean ± std | `run_experiments.aggregate`, `multiseed_summary.*` | 3 seeds reproduced byte-identically | PASS | Only 3 seeds; other P2 items not done |
