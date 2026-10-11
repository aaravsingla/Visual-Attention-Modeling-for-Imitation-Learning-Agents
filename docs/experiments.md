# Experiments

All numbers are copied from files written by `python scripts/reproduce_results.py --clean`
(`results/metrics/test_metrics_seed0.json`, `results/metrics/summary_seed0.csv`). The expert is
synthetic. Dwell times are synthetic milliseconds.

## E1. Behavior Cloning vs. baselines (held-out test documents, seed 0)

Setup: 100 documents × 5 trajectories; split by document 70/15/15; the test set has 15 documents and
8,857 transitions. Teacher-forced, one step ahead.

| model | dwell r | dwell MAE (ms) | dwell RMSE (ms) | dwell R² | move error (px) | move error (norm.) |
|---|---|---|---|---|---|---|
| constant (train means) | undefined | 38.7 | 48.6 | −0.006 | 88.1 | 0.146 |
| BC features only | 0.585 | 30.8 | 39.5 | 0.334 | 56.3 | 0.093 |
| BC shuffled-crop control | 0.583 | 30.9 | 40.0 | 0.319 | 57.0 | 0.094 |
| BC full | 0.671 | 28.1 | 36.0 | 0.447 | 31.7 | 0.052 |
| noise ceiling (reference) | 0.807 | 22.5 | 28.7 | 0.650 | – | – |

Validation split, same models: full r = 0.699 / MAE 27.4 ms / 30.1 px; features only 0.621 /
30.3 ms / 55.3 px; shuffled-crop control 0.624 / 30.0 ms / 56.0 px; constant MAE 39.1 ms / 87.8 px; noise
ceiling r = 0.813.

**Finding.** The saliency-conditioned BC policy is clearly better than every baseline. Its dwell
correlation (0.671) is below the 0.85 the resume line mentions and below the 0.807 noise ceiling.

## E2. Saliency ablation and shuffled-crop control

Two ways of taking the saliency information away, with the same features, training procedure,
splits and metrics:

* **features only:** no CNN and no saliency map/crop (25,731 parameters). The 7 scalar features,
  including a window-mean saliency value, stay.
* **shuffled crops:** exactly the full architecture (552,163 parameters) and training, but every
  transition is paired with the real crop of another, randomly chosen transition of the same split.

| metric (test) | full | features only | shuffled crops | change, full → features only |
|---|---|---|---|---|
| dwell MAE (ms) | 28.1 | 30.8 | 30.9 | +2.7 ms (worse) |
| dwell Pearson r | 0.671 | 0.585 | 0.583 | −0.086 |
| next-fixation error (px) | 31.7 | 56.3 | 57.0 | +24.6 px (worse) |
| return-sweep error (px) | 56.0 | 186.8 | 161.6 | +130.8 px |
| predicted next fixation on text | 92.1 % | 78.2 % | 74.3 % | −13.9 points |

(Differences are computed from the unrounded values in the metrics file.)

**Why.** The scalar features describe only a 40 x 16 px window. They contain contrast and ink density
(two of the three dwell factors) but only weak information about word length, the main factor (a
linear fit of word length on the 7 features gives R² = 0.28). They say nothing about where the
current line ends or where the next line starts. The saliency crop shows both. The shuffled-crop
control has the same network and real crop statistics, and it performs like the small feature-only
model. So the gain requires the crop *at the current gaze*. It does not come from the larger network
or from crop-like inputs. (A first version of this control fed an all-zero image. A constant
input only adds a bias, so it was equivalent to the feature-only model by construction and was
replaced after the second review.) The ablation therefore establishes that, *in this
simulator*, the layout and word-length information in the saliency crop is used, mainly for
movement. It does not establish that saliency explains human reading. The architecture was chosen
on validation for the full model only, which would, if anything, favour the full model.

## E3. Error analysis by saccade type (test, px)

| model | forward (6,559) | skip (715) | regression (560) | return sweep (1,023) |
|---|---|---|---|---|
| constant | 45.2 | 74.6 | 139.4 | 344.3 |
| features only | 29.8 | 39.3 | 148.9 | 186.8 |
| shuffled crops | 34.1 | 43.4 | 151.8 | 161.6 |
| full | 17.2 | 24.1 | 166.6 | 56.0 |

Regressions are random events in the expert (probability 4-11 %, jump length 1-3 words). The models
predict the conditional mean, which is close to a forward move, so they do worst on regressions,
even worse than the constant baseline. Skips are partly predictable (short words are skipped more
often), so the error there is moderate.

## E4. Free-running rollouts (test, mean over 15 documents)

The shuffled-crop control is not rolled out: its input is not tied to its own gaze.

| scanpath | fixations on text | words fixated | lines visited | steps clipped |
|---|---|---|---|---|
| synthetic expert | 98.4 % | 91.9 % | 100 % | – |
| BC full | 63.7 % | 60.8 % | 90.4 % | 17.8 % |
| BC features only | 35.9 % | 34.5 % | 89.7 % | 10.3 % |
| constant | 30.3 % | 11.4 % | 88.0 % | 0 % |

Qualitatively, the full policy reads left to right with return sweeps but often skips a line in dense
paragraphs. The first-40-fixation figures are `results/figures/scanpath_doc*_{expert,learned,comparison}.png`.
`scanpath_doc*_full_rollout.png` shows the whole rollout. Once the policy's gaze leaves the text
(after the last paragraph, or after a bad return sweep), it sees states never present in the
demonstrations and drifts to the bottom-left corner. With the final seed-0 checkpoint, 10 of the 15
full-model rollouts end exactly at (0, 1), and 11 end on the bottom edge. This is the classic BC compounding-error problem.
Teacher-forced metrics do not show it. "Lines visited" is a weak metric (a policy that drifts slowly
downward crosses most lines); "words fixated" separates the methods better.

## E5. Development runs (validation only)

See `docs/methodology.md`, section 8, and the raw logs in `docs/development_log.md`. Four changes
were tried, all judged on validation documents. The final configuration (no CNN pooling;
100 documents × 5 trajectories) was fixed before the test split was evaluated for the first time.

## E6. MaxEnt IRL reward recovery (`results/metrics/irl_results.json`)

Setup: 8 x 6 grid MDP per page, horizon 30. 30 training pages and 15 held-out pages (ids 1000-1044,
not used by BC). 20 demonstrations per page from the soft-optimal policy for
`w_true = [1.0, 2.0, −1.5, −0.5, −1.0]` (features: landing saliency, landing density, regress, skip,
sweep). MaxEnt IRL ran 500 Adam iterations (lr 0.05) from w = 0; the whole IRL experiment, including the sweep, took 49.6 s.

| feature | w_true | w (IRL) | θ (one-step BC logit, not a reward) |
|---|---|---|---|
| landing saliency | 1.000 | 0.902 | 1.975 |
| landing density | 2.000 | 2.001 | 2.072 |
| regress | −1.500 | −1.463 | −0.159 |
| skip | −0.500 | −0.522 | −1.197 |
| sweep | −1.000 | −1.071 | −1.354 |

Recovery: cosine 0.999, relative L2 error 0.044 (success criterion fixed in advance: ≥ 0.99 and
≤ 0.15). Training log-likelihood per trajectory rose from −35.54 (w = 0) to −24.554; at w_true it is
−24.560.

| policy (held-out pages) | NLL / step | TV to true policy | argmax agreement | expected true return |
|---|---|---|---|---|
| true soft-optimal | 0.8251 | 0 | 100 % | 54.82 |
| MaxEnt IRL | 0.8253 | 0.006 | 99.3 % | 54.68 |
| one-step BC logit | 0.9197 | 0.140 | 78.3 % | 44.56 |
| uniform legal actions | 1.2958 | 0.404 | 46.2 % | 7.65 |

| feature expectations per episode (held-out) | saliency | density | regress | skip | sweep |
|---|---|---|---|---|---|
| empirical expert demonstrations | 25.02 | 25.34 | 10.94 | 2.69 | 3.15 |
| true policy (exact) | 25.08 | 25.39 | 11.04 | 2.71 | 3.13 |
| IRL policy (exact) | 24.99 | 25.36 | 11.05 | 2.71 | 3.10 |
| one-step BC logit (exact) | 21.32 | 22.53 | 11.56 | 2.48 | 3.24 |

Data-size sweep (the first n of the 20 trajectories per training page, so nested subsets of one
draw; n = 20 is the main run → cosine / relative error / held-out NLL): 1 → 0.998 / 0.068 / 0.8253;
5 → 0.996 / 0.095 / 0.8258; 20 → 0.999 / 0.044 / 0.8253. There is one sample at each size, so the
ordering between 1 and 5 is noise. The pre-set success criterion is therefore lenient: even
1 trajectory per page meets it. The saliency weight has the largest error. Cell saliency and density
are strongly collinear (r = 0.88 on page 1030, 0.96 pooled over training cells).

**Finding.** Reward recovery succeeds in this controlled setting, and the derived policy matches the
expert on new pages. The one-step BC model fits the same demonstrations but cannot plan. Its
weights differ qualitatively from the reward: the regress weight is almost 0, because a myopic
model explains regressions by where they land rather than by their cost. Its held-out likelihood and
true return are clearly worse. This comparison is structurally favourable to IRL: IRL gets the exact
dynamics, horizon and start state, and the expert is inside its model class. It shows that a myopic
model is mis-specified for a planning expert, not that IRL is generally better. Limits: the expert is exactly in the model class, and its memoryless
reward makes it oscillate over dense cells, so the experiment validates the method, not a theory
of reading.

## E7. Training seeds (P2)

Produced by `python scripts/reproduce_results.py --clean --seeds 0 1 2 --copy-figures` (977.5 s of BC
training and evaluation for the three seeds; 1039.4 s in total including data and IRL). Data, splits
and test set are fixed; only the training seed changes.

| model | dwell r (mean ± std) | per seed | dwell MAE (ms) | move error (px) | per seed (px) |
|---|---|---|---|---|---|
| constant | undefined | – | 38.75 ± 0 | 88.08 ± 0 | – |
| features only | 0.592 ± 0.006 | 0.585 / 0.595 / 0.595 | 30.75 ± 0.09 | 56.52 ± 1.84 | 56.3 / 54.8 / 58.5 |
| shuffled crops | 0.590 ± 0.007 | 0.583 / 0.592 / 0.597 | 30.81 ± 0.22 | 56.22 ± 1.17 | 57.0 / 54.9 / 56.8 |
| full | 0.673 ± 0.006 | 0.671 / 0.667 / 0.680 | 28.19 ± 0.32 | 30.55 ± 0.97 | 31.7 / 30.0 / 29.9 |

Rollouts (mean ± std over seeds): fixations on text 73.0 ± 8.2 % (full) vs. 28.6 ± 8.0 % (features
only); words fixated 62.5 ± 1.9 % vs. 27.9 ± 6.5 %; steps clipped 10.7 ± 6.1 % vs. 14.6 ± 7.1 %.

**Finding.** The teacher-forced conclusions are stable across seeds. The seed-0 numbers used in E1-E4
are typical. Rollout metrics are much noisier, because small one-step differences compound.

## Not done

* No human eye-tracking data, so no human validation.
* No study of how the dwell correlation changes with the synthetic noise level. This would mainly
  show that the correlation can be set by the generator.
* IRL was not run on the rule-based P0 expert's demonstrations (no ground-truth reward exists for it,
  and its moves do not map cleanly onto the grid actions).
* P2 items other than multiple seeds (better saliency, more IRL features, interactive visualisation).
