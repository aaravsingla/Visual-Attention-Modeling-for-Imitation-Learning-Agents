"""Synthetic text-document images with known layout.

Each document is a grayscale page (default 600 x 800 px, white background) with a heading
and several paragraphs of pseudo-words. Paragraphs differ in font size, line spacing,
word spacing, line width, text gray level (contrast) and optional background tint.
Pseudo-words are random letter strings, so there is no semantic content: any "reading"
behaviour in this project is driven only by layout and visual features.

Pixel coordinates: x = column (0 = left), y = row (0 = top).
Every document is generated from ``np.random.default_rng([seed, doc_id])`` so a given
(seed, doc_id) pair always produces the same page.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "etaoinshrdlcumwfgypbvkjxqz"
# Approximate English letter frequencies (same order as LETTERS), for a text-like texture.
_FREQ = np.array([12.7, 9.1, 8.2, 7.5, 7.0, 6.7, 6.3, 6.1, 6.0, 4.3, 4.0, 2.8, 2.8, 2.4,
                  2.4, 2.2, 2.0, 2.0, 1.9, 1.5, 1.0, 0.8, 0.15, 0.15, 0.1, 0.07])
LETTER_P = _FREQ / _FREQ.sum()


@dataclass
class Word:
    text: str
    box: list[int]          # [x0, y0, x1, y1] in pixels (tight ink box)
    line: int               # index into Document.lines
    contrast: float         # (background gray - text gray) / 255, in (0, 1]


@dataclass
class Line:
    box: list[int]          # [x0, y0, x1, y1] union of word boxes
    paragraph: int          # 0 = heading
    is_heading: bool
    font_size: int
    style: str
    word_ids: list[int] = field(default_factory=list)


@dataclass
class Document:
    doc_id: int
    width: int
    height: int
    image: np.ndarray       # (height, width) uint8, 255 = white
    words: list[Word]
    lines: list[Line]

    @property
    def n_chars(self) -> np.ndarray:
        return np.array([len(w.text) for w in self.words])

    def layout_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "width": self.width,
            "height": self.height,
            "words": [asdict(w) for w in self.words],
            "lines": [asdict(l) for l in self.lines],
        }

    def save(self, raw_dir: str | Path) -> None:
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.image).save(raw_dir / f"doc_{self.doc_id:04d}.png")
        with open(raw_dir / f"doc_{self.doc_id:04d}.json", "w", encoding="utf-8") as f:
            json.dump(self.layout_dict(), f)


def load_document(raw_dir: str | Path, doc_id: int) -> Document:
    raw_dir = Path(raw_dir)
    image = np.array(Image.open(raw_dir / f"doc_{doc_id:04d}.png").convert("L"))
    with open(raw_dir / f"doc_{doc_id:04d}.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return Document(
        doc_id=d["doc_id"], width=d["width"], height=d["height"], image=image,
        words=[Word(**w) for w in d["words"]], lines=[Line(**l) for l in d["lines"]],
    )


def _uniform_int(rng: np.random.Generator, lo_hi) -> int:
    lo, hi = lo_hi
    return int(rng.integers(lo, hi + 1))


def _uniform(rng: np.random.Generator, lo_hi) -> float:
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def _pseudo_word(rng: np.random.Generator, cfg: dict) -> str:
    lo, hi = cfg["word_length"]
    n = int(np.clip(rng.poisson(cfg["word_length_mean"] - 1) + 1, lo, hi))
    return "".join(rng.choice(list(LETTERS), size=n, p=LETTER_P))


def generate_document(doc_id: int, cfg: dict, seed: int) -> Document:
    """Render one synthetic document. ``cfg`` is the ``document`` config section."""
    rng = np.random.default_rng([seed, doc_id])
    W, H, m = cfg["width"], cfg["height"], cfg["margin"]
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)

    words: list[Word] = []
    lines: list[Line] = []
    # Layout is computed first; background tints are drawn before the text afterwards.
    planned: list[tuple] = []                # (x, y, text, font, gray, stroke) per word
    tints: list[tuple[int, int, int]] = []   # (top, bottom, gray) background boxes

    def layout_line(font, y, max_width, word_gap, para, is_heading, size, style,
                    gray, bg, stroke, n_words=None):
        """Place words left to right starting at y until max_width (or n_words) is reached."""
        x = m
        line_idx = len(lines)
        line = Line(box=[0, 0, 0, 0], paragraph=para, is_heading=is_heading,
                    font_size=size, style=style)
        boxes = []
        while True:
            if n_words is not None and len(line.word_ids) >= n_words:
                break
            text = _pseudo_word(rng, cfg)
            if is_heading:
                text = text.capitalize()
            box = draw.textbbox((x, y), text, font=font, stroke_width=stroke)
            if box[2] > m + max_width and line.word_ids:
                break
            if box[2] > W - m:  # a single over-long word: stop the line
                break
            contrast = (bg - gray) / 255.0
            line.word_ids.append(len(words))
            words.append(Word(text=text, box=[int(v) for v in box], line=line_idx,
                              contrast=round(contrast, 4)))
            planned.append((x, y, text, font, gray, stroke))
            boxes.append(box)
            x = box[2] + word_gap
        if not boxes:
            return None
        b = np.array(boxes)
        line.box = [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())]
        lines.append(line)
        return line

    # --- heading -------------------------------------------------------------------
    y = m
    hsize = _uniform_int(rng, cfg["heading_font_size"])
    hfont = ImageFont.load_default(size=hsize)
    hgray = _uniform_int(rng, cfg["text_gray_high"])
    layout_line(hfont, y, W - 2 * m, int(0.35 * hsize), 0, True, hsize, "heading",
                hgray, 255, 1, n_words=_uniform_int(rng, cfg["heading_words"]))
    y += int(hsize * 1.6) + _uniform_int(rng, cfg["paragraph_gap"])

    # --- paragraphs -------------------------------------------------------------------
    styles = list(cfg["style_probs"].keys())
    probs = np.array([cfg["style_probs"][s] for s in styles], dtype=float)
    n_par = _uniform_int(rng, cfg["n_paragraphs"])
    for p in range(1, n_par + 1):
        style = styles[rng.choice(len(styles), p=probs / probs.sum())]
        st = cfg["styles"][style]
        size = _uniform_int(rng, st["font_size"])
        font = ImageFont.load_default(size=size)
        line_h = int(round(size * st["line_spacing"]))
        gap = _uniform_int(rng, st["word_gap"])
        width = int((W - 2 * m) * _uniform(rng, st["width_frac"]))
        tinted = rng.random() < cfg["p_tinted_background"]
        bg = _uniform_int(rng, cfg["tint_gray"]) if tinted else 255
        low = rng.random() < cfg["p_low_contrast"]
        gray = _uniform_int(rng, cfg["text_gray_low"] if low else cfg["text_gray_high"])
        if tinted:  # keep text visibly darker than the tint
            gray = min(gray, bg - 70)
        n_lines = _uniform_int(rng, cfg["lines_per_paragraph"])
        top = y
        placed = 0
        for li in range(n_lines):
            if y + line_h > H - m:
                break
            last = li == n_lines - 1
            w = int(width * rng.uniform(0.3, 0.9)) if last else width
            if layout_line(font, y, w, gap, p, False, size, style, gray, bg, 0) is not None:
                placed += 1
            y += line_h
        if placed and tinted:
            tints.append((top - 6, y - line_h + size + 8, bg))
        y += _uniform_int(rng, cfg["paragraph_gap"])
        if y > H - m - 20:
            break

    for top, bottom, bg in tints:
        draw.rectangle([m - 10, top, W - m + 10, bottom], fill=bg)
    for x, yy, text, font, gray, stroke in planned:
        draw.text((x, yy), text, fill=gray, font=font, stroke_width=stroke, stroke_fill=gray)

    return Document(doc_id=doc_id, width=W, height=H, image=np.array(img, dtype=np.uint8),
                    words=words, lines=lines)


def generate_documents(n: int, cfg: dict, seed: int) -> list[Document]:
    return [generate_document(i, cfg, seed) for i in range(n)]
