"""Transparent rule-based synthetic reading expert.

The expert knows the document layout (word boxes in reading order). Its decision rule at
each fixation on word ``i``:

1. If ``i`` is the last word of the document (or ``max_fixations`` is reached): stop.
2. **Regression** with probability ``p_regress_base + p_regress_low_contrast * (1 - contrast_i)``
   (if ``i > 0``): jump back ``k ~ U{1..regress_max_words}`` words, but not before the first
   word of the previous line. Reading then continues forward from there.
3. Otherwise, if ``i`` is the last word on its line: **return sweep** to the first word of the
   next line.
4. Otherwise move to the next word ``i+1``, or **skip** it (land on ``i+2``, same line only)
   with probability ``p_skip * (1 - skip_saliency_factor * saliency_{i+1})`` where ``p_skip``
   is ``p_skip_short`` for words of at most ``short_word_chars`` characters and
   ``p_skip_long`` otherwise.

Landing position on the target word: ``x = x0 + clip(N(landing_mean, landing_std), 0, 1) * width``
and ``y = line centre + N(0, vertical_std * line height)``, then normalised to ``[0, 1]``.

Dwell time (synthetic milliseconds) of a fixation on word ``i``::

    clean_i = base + per_char * n_chars_i + density_ms * density_i + low_contrast_ms * (1 - contrast_i)
    dwell_i = max(min_ms, clean_i * exp(N(0, noise_sigma)))

``density_i`` is the mean ink density (saliency module) inside the word's box, and
``contrast_i`` is the word's text contrast. The noise is not observable by any model, so the
correlation between ``clean`` and ``dwell`` is an upper bound ("noise ceiling") on what a
predictor can reach.
"""

from __future__ import annotations

import numpy as np

from .document_generator import Document
from .saliency import SaliencyMaps

EVENT_NAMES = ("forward", "skip", "regression", "return_sweep")
FORWARD, SKIP, REGRESSION, RETURN_SWEEP = range(4)


class ReadingExpert:
    def __init__(self, doc: Document, maps: SaliencyMaps, cfg_expert: dict):
        self.doc = doc
        self.cfg = cfg_expert
        self.W, self.H = doc.width, doc.height
        self.n_words = len(doc.words)
        self.boxes = np.array([w.box for w in doc.words], dtype=np.float64).reshape(-1, 4)
        self.n_chars = doc.n_chars
        self.contrast = np.array([w.contrast for w in doc.words], dtype=np.float64)
        self.line_of = np.array([w.line for w in doc.words], dtype=int)
        self.line_first = np.array([doc.lines[l].word_ids[0] for l in self.line_of])
        self.line_last = np.array([doc.lines[l].word_ids[-1] for l in self.line_of])
        self.line_box = np.array([doc.lines[l].box for l in self.line_of], dtype=np.float64).reshape(-1, 4)
        self.word_saliency = self._box_mean(maps.saliency)
        self.word_density = self._box_mean(maps.density)
        d = cfg_expert["dwell"]
        self.clean_dwell = (d["base_ms"] + d["per_char_ms"] * self.n_chars
                            + d["density_ms"] * self.word_density
                            + d["low_contrast_ms"] * (1.0 - self.contrast))

    def _box_mean(self, m: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_words)
        for i, (x0, y0, x1, y1) in enumerate(self.boxes.astype(int)):
            out[i] = m[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].mean()
        return out

    def landing(self, i: int, rng: np.random.Generator) -> np.ndarray:
        x0, _, x1, _ = self.boxes[i]
        _, ly0, _, ly1 = self.line_box[i]
        frac = np.clip(rng.normal(self.cfg["landing_mean"], self.cfg["landing_std"]), 0.0, 1.0)
        x = x0 + frac * (x1 - x0)
        y = 0.5 * (ly0 + ly1) + rng.normal(0.0, self.cfg["vertical_std"] * (ly1 - ly0))
        return np.clip(np.array([x / self.W, y / self.H]), 0.0, 1.0)

    def dwell(self, i: int, rng: np.random.Generator) -> tuple[float, float]:
        d = self.cfg["dwell"]
        clean = float(self.clean_dwell[i])
        noisy = max(d["min_ms"], clean * float(np.exp(rng.normal(0.0, d["noise_sigma"]))))
        return noisy, clean

    def next_word(self, i: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        """Apply the decision rule. Returns (next word index, event) or (None, None) to stop."""
        if i >= self.n_words - 1:
            return None, None
        c = self.cfg
        p_reg = c["p_regress_base"] + c["p_regress_low_contrast"] * (1.0 - self.contrast[i])
        if i > 0 and rng.random() < p_reg:
            k = int(rng.integers(1, c["regress_max_words"] + 1))
            line = self.line_of[i]
            floor = self.doc.lines[line - 1].word_ids[0] if line > 0 else 0
            return max(i - k, floor), REGRESSION
        if i == self.line_last[i]:
            return i + 1, RETURN_SWEEP
        j = i + 1
        p_skip = c["p_skip_short"] if self.n_chars[j] <= c["short_word_chars"] else c["p_skip_long"]
        p_skip *= 1.0 - c["skip_saliency_factor"] * self.word_saliency[j]
        if j + 1 <= self.line_last[i] and rng.random() < p_skip:
            return j + 1, SKIP
        return j, FORWARD

    def sample_trajectory(self, rng: np.random.Generator) -> dict:
        """One reading trajectory starting at the first word.

        Returns arrays: ``fixations (T, 2)``, ``dwell (T,)``, ``dwell_clean (T,)``,
        ``word (T,)`` and ``event (T-1,)`` (event of the saccade from fixation t to t+1).
        """
        i = 0
        fix, dwell, clean, words, events = [], [], [], [], []
        while True:
            fix.append(self.landing(i, rng))
            d, dc = self.dwell(i, rng)
            dwell.append(d)
            clean.append(dc)
            words.append(i)
            if len(fix) >= self.cfg["max_fixations"]:
                break
            j, ev = self.next_word(i, rng)
            if j is None:
                break
            events.append(ev)
            i = j
        return {"fixations": np.array(fix), "dwell": np.array(dwell), "dwell_clean": np.array(clean),
                "word": np.array(words), "event": np.array(events, dtype=int)}
