# Interview notes

Short, code-consistent answers. Numbers are from the seed-0 run (`results/metrics/test_metrics_seed0.json`).
Format per topic: **Intuition**, **Technical idea**, **Likely follow-up**, **Caveat**.

---

### 1. What problem does the project address?
* **Intuition:** Can a small network, looking only at what is around the current gaze point, imitate
  where a reader looks next and how long they look?
* **Technical:** Imitation learning of a two-part action (saccade displacement, fixation duration) from
  demonstrations of a synthetic reading expert on synthetic text pages.
* **Follow-up:** "Why not real data?" No eye-tracker or licensed dataset was available, and a known
  expert lets us control and audit the experiment.
* **Caveat:** It is a simulation. Results are about imitating *this* expert.

### 2. What does a saliency map represent?
* **Intuition:** A heat map of what visually stands out (high contrast, strong edges, e.g. the bold heading).
* **Technical:** `minmax(Gauss_σ4(0.5·localstd/127.5 + 0.5·|Sobel|/1020))` in [0, 1]. Ink density and
  the text mask are separate maps.
* **Follow-up:** "Why not a pretrained saliency model?" Not needed for the question, it would add a
  download and a dependency, and the spec asked for a deterministic offline pipeline.
* **Caveat:** It is a proxy for prominence, not measured human attention.

### 3. State and action spaces
* **Intuition:** The state is "what I see around my gaze and where I am". The action is "where to
  jump and how long to look".
* **Technical:** State = `1x84x84` saliency crop (300 x 200 px window centred on the gaze) + 7 features
  `[x, y, saliency, density, contrast, edges, text_occupancy]`. Action = `(dx, dy)` in normalised
  page units and dwell in synthetic ms. Transition `g' = clip(g + d, 0, 1)`.
* **Follow-up:** "Is the state Markov?" Not quite. The expert's decision depends on its word index and
  random draws. The crop only approximately identifies the word, and nothing in the state says
  whether a regression just happened.
* **Caveat:** Continuous action regressed with MSE; no action distribution is modelled.

### 4. How does the synthetic expert generate trajectories?
* **Intuition:** It reads word by word, sometimes skips short words, sometimes jumps back, and sweeps
  to the next line at line ends.
* **Technical:** Regression probability `0.04 + 0.10(1 − contrast)`. Skip probability `0.45` (≤ 3
  letters) or `0.04`, times `(1 − 0.5·saliency)`. Landing point 40 % ± 12 % into the word. Measured
  event mix on the test set: 74 % forward, 8 % skip, 6 % regression, 12 % return sweep.
* **Follow-up:** "Is that human-like?" Loosely inspired by well-known reading phenomena (skipping of
  short words, regressions, return sweeps), but none of the parameters are fitted to human data.
* **Caveat:** The expert sees exact word boxes; the model sees only pixels.

### 5. How are the dwell targets constructed?
* **Intuition:** Longer words, words with more ink per area (larger or bolder fonts) and fainter words
  get longer fixations, plus random noise.
* **Technical:** `clean = 140 + 18·letters + 120·density + 100·(1 − contrast)`;
  `dwell = max(80, clean·exp(N(0, 0.1²)))`. The noise is unobservable, so `corr(clean, dwell)` = 0.807
  on the test set is a ceiling.
* **Follow-up:** "Why multiplicative log-normal noise?" It keeps durations positive and right-skewed,
  like real fixation durations.
* **Caveat:** Changing σ changes every dwell metric. The correlation is a property of the generator
  as much as of the model.

### 6. How is the BC loss calculated?
* **Intuition:** Squared error on movement plus squared error on dwell, after putting both on the same scale.
* **Technical:** `L = λm·MSE(ẑm, (d − μd)/σd) + λw·MSE(ẑw, (log w − μlog)/σlog)` with λ = 1, and
  statistics from the training split. Prediction: `w = exp(ẑw·σlog + μlog)`.
* **Follow-up:** "What does MSE on log-dwell estimate?" The conditional mean of log-dwell, so the
  exponentiated prediction is closer to a conditional median/geometric mean than to the mean.
* **Caveat:** MSE averages over random choices (skip vs. no skip), so predictions can fall between modes.

### 7. Why separate movement and dwell heads?
* **Intuition:** "How long" and "where next" are different questions.
* **Technical:** A shared encoder and trunk feed two linear heads. Targets are z-scored separately,
  so neither loss dominates. Dwell depends mostly on the fixated word; movement depends on the
  surrounding layout.
* **Follow-up:** "Would two separate networks be better?" Possibly for dwell (sharing could interfere).
  Not tested.
* **Caveat:** The λ weights were not tuned (both 1).

### 8. Why does train/test leakage matter here?
* **Intuition:** The same page appears in all 5 trajectories of a document. If its transitions were
  in both train and test, the model could memorise the words of the test pages.
* **Technical:** The split is by document id (70/15/15), saved to `splits.json`, and checked to be
  disjoint on every load. Normalisation statistics and the constant baseline use training data only.
  The test split was not evaluated until the configuration was fixed on validation.
* **Follow-up:** "How do you know the inputs don't leak future actions?" Features are recomputed from
  the image at the *current* gaze in a unit test, and the crop is a function of image + current gaze only.
* **Caveat:** Training and test pages come from the same generator, so this is in-distribution generalisation.

### 9. Pearson correlation vs. MAE
* **Intuition:** r asks "do predictions go up and down with the targets?"; MAE asks "how far off are
  they, in ms?"
* **Technical:** r is invariant to shifting and scaling the predictions; MAE is not. A prediction
  `2·w − 280` can have r = 1 and a large MAE. The constant baseline has undefined r but a finite MAE (38.7 ms).
* **Follow-up:** "Is r = 0.67 '67 % accuracy'?" No. r² ≈ 0.45 is the share of variance explained
  by a linear fit.
* **Caveat:** r is sensitive to the target range; a narrower dwell range would lower r for the same errors.

### 10. What does the saliency ablation establish?
* **Intuition:** Remove the map and see what breaks.
* **Technical:** The ablation model is built without the CNN (no saliency map/crop; the scalar features,
  including a window-mean saliency value, stay). Test:
  movement error 31.7 → 56.3 px (return sweeps 56.0 → 186.8 px), dwell r 0.671 → 0.585,
  MAE 28.1 → 30.8 ms.
* **Follow-up:** "Is it the saliency, or just more parameters?" There is a control for this: the full
  552k-parameter network trained on real crops taken from *other* random transitions. It does no
  better than the 26k-parameter feature-only model (r 0.583, 57.0 px), so the gain needs the crop at
  the current gaze. (A first version used an all-zero image. That is useless as a control, because a
  constant input only adds a bias. Know why if asked.)
* **Caveat:** It shows that the crop carries layout and word-length information in this simulator, not
  that visual saliency drives human reading.

### 11. What are saccades in the simulation?
* **Intuition:** The arrows between fixations.
* **Technical:** A saccade is the displacement `g_{t+1} − g_t`. It is instantaneous; there is no
  saccade duration, velocity profile or saccadic suppression.
* **Follow-up:** "Do you model the perceptual span?" Only implicitly, through the size of the crop
  window (300 x 200 px).
* **Caveat:** A computational analogy, not oculomotor modelling.

### 12. Behavior Cloning vs. IRL
* **Intuition:** BC copies actions; IRL infers *why* (a reward) and then plans with it.
* **Technical:** BC: `min_θ E[ℓ(π_θ(s), a_expert)]`. IRL: find `w` such that the policy that is optimal
  for `R = wᵀφ` reproduces the expert's feature expectations, then derive the policy from `R`.
* **Follow-up:** "Is your neural model an IRL agent?" No. The neural model is pure BC. IRL is a
  separate experiment (`irl.py`) on a small grid MDP. There, IRL (held-out NLL 0.8253 per step) beats a
  one-step BC model on the same features (0.9197), because the expert plans ahead and BC is myopic. That comparison is built to favour IRL (exact dynamics, expert inside the model class), so it shows BC mis-specification, not a general IRL advantage.
* **Caveat:** IRL needs a model of the dynamics and repeated planning in the inner loop; that is why it
  is done on a 48-state grid and not on the continuous gaze environment.

### 13. How is the IRL reward inferred?
* **Intuition:** Find reward weights under which a soft-optimal agent would visit the same kinds of
  cells and make the same kinds of moves, on average, as the expert.
* **Technical:** MaxEnt IRL (Ziebart et al., 2008). With deterministic dynamics,
  `log P(τ|w) = wᵀΦ(τ) − V_0(s_0)`, which is concave. Gradient = expert feature counts − expected
  feature counts of the soft-optimal policy for the current `w` (exact forward pass over visitation
  frequencies), followed by Adam ascent. Measured: `w_true = [1, 2, −1.5, −0.5, −1]`, recovered
  `[0.90, 2.00, −1.46, −0.52, −1.07]` from 600 trajectories on 30 documents; cosine 0.999, relative L2
  error 0.044.
* **Follow-up:** "Is that a fair test?" It is a *controlled* test: the expert is exactly a MaxEnt agent
  with a reward inside the model class, so recovery is expected if the code is right. It checks the
  implementation; it does not show that readers optimise this reward. Also: "Why can't you run it on
  the P0 expert?" That expert is rule-based and has no reward, so there would be nothing to compare against.
* **Caveat:** Rewards are only identifiable up to transformations that leave behaviour unchanged
  (e.g. potential-based shaping). The grid expert is not a realistic reader: a memoryless reward makes
  it oscillate over dense cells (about 11 regressions per 30 steps).

### 14. Teacher-forced evaluation vs. free-running rollout
* **Intuition:** Teacher forcing tests "given the expert's position, what next?". A rollout tests
  "left on its own, where does the model go?".
* **Technical:** Teacher-forced: one-step metrics over all test transitions. Rollout: start at the
  expert's first fixation and use the model's own displacements for as many steps as the expert took.
  The full model keeps 64 % of rollout fixations on text vs. 98 % for the expert, and 18 % of steps
  are clipped at the border.
* **Follow-up:** "How would you fix the drift?" DAgger-style data aggregation (query the expert on
  states the policy visits), adding off-text recovery states to the demonstrations, or a stopping action.
* **Caveat:** Rollout metrics depend on the step budget, which here equals the expert's trajectory length.

### 15. Limitations of synthetic data
* **Intuition:** We built the teacher, so success shows that the student can learn *our* teacher, not
  how people read.
* **Technical:** Targets come from known rules, so the task is only as hard as the rules make it, and the
  metrics depend on chosen noise levels (σ = 0.10 → noise ceiling r = 0.807). Pseudo-words remove all
  linguistic effects (word frequency, predictability) that drive real dwell times. The expert has
  exact layout knowledge; real readers don't. The 15 test documents come from the same generator, so
  this tests in-distribution generalisation only.
* **Likely follow-up:** "So what is the value?" A controlled, leakage-free testbed with a known ceiling,
  where architecture choices (e.g. the saliency crop) and evaluation protocols (teacher-forced vs.
  rollout) can be checked before spending effort on real data.
* **Caveat:** Results can be made better or worse by changing the generator. Only comparisons inside
  one fixed generator are meaningful.

### 16. Next experiment with real eye-tracking data
* **Intuition:** Replace the synthetic teacher with real readers and keep everything else.
* **Technical:** Take an existing public reading eye-tracking corpus with on-screen text and recorded
  fixations. Render the same text as images, compute the same saliency and features, and train the same
  two-headed policy on human fixations with a *participant- and text-disjoint* split. Compare against
  the same baselines plus a word-length/frequency regression. Report r, MAE and next-fixation error,
  and add a scanpath-similarity measure for rollouts.
* **Likely follow-up:** "What noise ceiling would you use?" Agreement between readers, e.g. how well
  one group of participants' mean dwell per word predicts another group's (split-half).
* **Caveat:** Human data brings inter-subject variability and linguistic effects that the current
  features cannot see, so expect lower correlations. Be ready to say that this has not been done.
