"""Deterministic handcrafted saliency and text-density maps (OpenCV + NumPy, no learned model).

All maps have the same height x width as the input image and are float32 in [0, 1].

* ``ink``        binary: pixel is darker than its local mean by ``ink_offset`` gray levels
                 (adaptive threshold, so light text on a tinted box is still found).
* ``density``    local fraction of ink pixels in a ``density_window`` box. Absolute scale.
* ``text_mask``  binary: ink closed with a wide kernel so letters merge into text regions.
* ``contrast``   local intensity standard deviation / 127.5 (127.5 is the maximum possible).
* ``edges``      Sobel gradient magnitude / (4 * 255), Gaussian-smoothed. Absolute scale.
* ``saliency``   blurred weighted sum of contrast and edges, min-max normalised per image.

Saliency is a proxy for *visual prominence* (high-contrast, edge-rich regions such as the
bold heading). It is not a model of human gaze, and it is kept separate from ``density``
(how much ink there is), which is closer to "how much there is to read".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

EPS = 1e-8


@dataclass
class SaliencyMaps:
    saliency: np.ndarray
    density: np.ndarray
    text_mask: np.ndarray
    contrast: np.ndarray
    edges: np.ndarray
    ink: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.saliency.shape


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _to_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def compute_maps(image: np.ndarray, cfg: dict) -> SaliencyMaps:
    """Compute all feature maps for a grayscale (or RGB) uint8 image. ``cfg`` = ``saliency`` section."""
    gray = _to_gray(image)
    g = gray.astype(np.float32)

    ink = cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                cfg["ink_block_size"], cfg["ink_offset"]).astype(np.float32)
    k = cfg["density_window"]
    density = cv2.boxFilter(ink, -1, (k, k), normalize=True, borderType=cv2.BORDER_CONSTANT)

    kw, kh = cfg["text_region_kernel"]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    text_mask = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
    text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    s = cfg["contrast_sigma"]
    mean = cv2.GaussianBlur(g, (0, 0), s)
    sq = cv2.GaussianBlur(g * g, (0, 0), s)
    contrast = np.sqrt(np.maximum(sq - mean * mean, 0.0)) / 127.5

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy) / (4.0 * 255.0), (0, 0), cfg["edge_sigma"])

    combo = cfg["contrast_weight"] * contrast + cfg["edge_weight"] * edges
    saliency = _minmax(cv2.GaussianBlur(combo, (0, 0), cfg["saliency_sigma"]))

    clip = lambda x: np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0).astype(np.float32)
    return SaliencyMaps(saliency=clip(saliency), density=clip(density), text_mask=clip(text_mask),
                        contrast=clip(contrast), edges=clip(edges), ink=clip(ink))


def downsample(m: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Area-average resize to ``size`` (int -> square, or (height, width))."""
    h, w = (size, size) if isinstance(size, int) else size
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


def overlay(image: np.ndarray, saliency: np.ndarray, ax=None, alpha: float = 0.5, title: str | None = None):
    """Draw ``saliency`` (any resolution) as a heat map over the document image."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6.5))
    h, w = _to_gray(image).shape
    sal = cv2.resize(saliency.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    # Transparent where saliency is low, so the page stays readable under the heat map.
    rgba = plt.get_cmap("inferno")(np.clip(sal, 0, 1))
    rgba[..., 3] = alpha * np.clip(sal / 0.3, 0, 1)
    im = ax.imshow(rgba)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax, im
