# Development log (validation split only)

These are the raw outputs of the development runs that led to the final configuration. They were
produced with a throw-away helper that trained the models and then called
`evaluate(cfg, splits_to_eval=("val",))`. **No test-split numbers were computed in any of these
runs.** The same validation-only scoring is available from the CLI:
`python -m visual_attention.evaluate --splits val`.

The helper wrote to a scratch directory outside the repository, so these lines are copied here by
hand. Except for the noted overrides, all runs used the configuration at the time (seed 0).
The expert rules, the dwell rule and its noise level (σ = 0.10), and the metric definitions were never changed.
At the time of these runs no control model existed yet. Runs v0, A and B used 60 documents, and
C and D used 100 and 200 documents. Their validation sets therefore differ (n = 7,397 / 9,796 / 10,644
transitions), so comparisons across those groups are only indicative. The helper is now in the
repository as `scripts/dev_validation_run.py`.

## v0: 60 documents × 6 trajectories, CNN with adaptive average pooling to 4 x 4

```
[val] n=7397 full: r=0.646 MAE=29.3ms move=33.3px | no_saliency: r=0.621 MAE=30.2ms move=54.8px | constant: r=undefined MAE=39.7ms move=83.5px | noise ceiling r=0.818
```

Full model stopped early at epoch 17 (best epoch 12).

## A: v0 without pooling (flatten the 11 x 11 conv map), kept

```
A full best epoch 6 val 0.8012 sec 152.6
A no_saliency best epoch 13 val 1.126 sec 17.2
A [val] n=7397 full: r=0.650 MAE=29.2ms move=31.6px | no_saliency: r=0.621 MAE=30.2ms move=54.8px | constant: r=undefined MAE=39.7ms move=83.5px | noise ceiling r=0.818
```

## B: A + saliency blur σ 4 → 1.5 (`saliency.saliency_sigma=1.5`), not kept

The skip rule uses saliency, so the expert trajectories (and n) changed slightly.

```
B full best epoch 9 val 0.7932 sec 178.5
B no_saliency best epoch 15 val 1.1318 sec 6.5
B [val] n=7426 full: r=0.649 MAE=29.6ms move=30.7px | no_saliency: r=0.620 MAE=30.3ms move=55.1px | constant: r=undefined MAE=39.6ms move=83.1px | noise ceiling r=0.816
```

## C: A + 100 documents × 5 trajectories, kept (final)

```
C full best epoch 6 val 0.7077 sec 198.4
C no_saliency best epoch 7 val 1.1842 sec 15.9
C [val] n=9796 full: r=0.701 MAE=27.4ms move=30.2px | no_saliency: r=0.621 MAE=30.3ms move=55.4px | constant: r=undefined MAE=39.1ms move=87.8px | noise ceiling r=0.813
```

## D: A + 200 documents × 3 trajectories, not kept

Outside the spec's suggested range of 40-100 documents, and only a small further gain.

```
D full best epoch 7 val 0.6779 sec 236.4
D no_saliency best epoch 22 val 1.1716 sec 17.5
D [val] n=10644 full: r=0.707 MAE=27.4ms move=28.2px | no_saliency: r=0.611 MAE=30.5ms move=53.2px | constant: r=undefined MAE=39.3ms move=86.7px | noise ceiling r=0.810
```

## After the configuration was fixed

* The first test-set evaluation followed configuration C.
* After the first independent review, three changes were made that do not tune anything:
  * a float32 consistency fix for the stored gaze,
  * evaluation now uses each checkpoint's stored normaliser,
  * a zero-input CNN capacity control was added.
* After the second independent review:
  * the zero-input control was replaced by the shuffled-crop control (a constant input only adds a
    bias, so it could not test capacity),
  * validation-only evaluation now writes separate files,
  * the IRL visitation figure uses one shared colour scale,
  * `data_seed` was separated from the training `seed`,
  * documentation was corrected.
  None of these changes alters a reported number other than adding the new control's numbers.

## IRL (P1)

`w_true = [1.0, 2.0, −1.5, −0.5, −1.0]`, the grid (8 x 6, T = 30), the page counts (30/15) and 20
trajectories per page were in the config from Stage 2. The optimiser placeholders written then
(lr 0.1, 300 iterations) were changed to lr 0.05 and 500 iterations before the first IRL run. That
change and the first run are in the same commit (f50d419), so git cannot prove the order; this
statement is the author's record. The success criterion (cosine ≥ 0.99 and relative L2 error ≤ 0.15)
was written into the code before the first run. The first run met it. Afterwards only the figure
(axis limits, later a shared colour scale) was changed. The criterion is lenient: even 1 trajectory
per page meets it.

After seeing that the grid expert makes about 11 regressions per 30 steps, the expert's behaviour
alone (no IRL run) was checked for other regress weights:

```
w_regress  regressions / skips / sweeps per 30 steps   cells with > 0.5 expected visits
  -1.5            11.4 / 2.7 / 2.7                           11.2
  -2.5             9.5 / 2.4 / 3.3                           13.6
  -3.0             8.3 / 2.4 / 3.6                           15.6
  -4.0             6.2 / 2.4 / 4.2                           18.9
```

The oscillation is structural (the reward has no memory), so `w_true` was left at its pre-specified
value.
