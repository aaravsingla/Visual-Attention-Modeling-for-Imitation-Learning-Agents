"""Lightweight reading environment: gaze state, observations and transitions.

Coordinate system
-----------------
Gaze is a normalised point ``(x, y)`` in ``[0, 1]^2``: ``x = column / width``,
``y = row / height``, origin at the top-left corner of the page. Pixels are used only
for drawing (``to_pixels``).

Action
------
A displacement ``(dx, dy)`` in the same normalised units plus a dwell time (synthetic
milliseconds) for the *current* fixation. The next gaze is ``clip(gaze + (dx, dy), 0, 1)``.

Episode
-------
``reset(gaze)`` places the first fixation. Each ``step`` records the dwell of the current
fixation and moves the gaze. The episode is done after ``max_steps`` steps.

Observation
-----------
``saliency``: ``(1, S, S)`` float32 crop of the saliency map centred on the gaze
(``S = crop_size``). The crop covers ``crop_width`` x ``crop_height`` of the page;
outside the page it is zero. ``features``: ``(F,)`` float32 vector, see ``FEATURE_NAMES``.
The observation depends only on the page and the current gaze - never on the expert's
future actions.
"""

from __future__ import annotations

import cv2
import numpy as np

from .saliency import SaliencyMaps, compute_maps

FEATURE_NAMES = ("x", "y", "saliency", "density", "contrast", "edges", "text_occupancy")


def clip_gaze(gaze) -> np.ndarray:
    return np.clip(np.asarray(gaze, dtype=np.float64), 0.0, 1.0)


def to_pixels(gaze, width: int, height: int) -> np.ndarray:
    g = np.asarray(gaze, dtype=np.float64)
    return g * np.array([width, height])


def _crop_pad(cfg_obs: dict) -> int:
    return cfg_obs["crop_size"] // 2 + 2


def observation_map(saliency: np.ndarray, cfg_obs: dict) -> np.ndarray:
    """Resample the page saliency so that the crop window becomes ``crop_size`` pixels, then zero-pad.

    The scale is anisotropic (the window is ``crop_width*W`` x ``crop_height*H`` page pixels).
    """
    H, W = saliency.shape
    cs = cfg_obs["crop_size"]
    out_w = int(round(cs / cfg_obs["crop_width"]))
    out_h = int(round(cs / cfg_obs["crop_height"]))
    small = cv2.resize(saliency.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.pad(small, _crop_pad(cfg_obs), mode="constant").astype(np.float32)


def extract_crop(obs_map: np.ndarray, gaze, cfg_obs: dict) -> np.ndarray:
    """``(crop_size, crop_size)`` saliency window centred on ``gaze`` (bilinear, sub-pixel)."""
    cs = cfg_obs["crop_size"]
    pad = _crop_pad(cfg_obs)
    out_h, out_w = obs_map.shape[0] - 2 * pad, obs_map.shape[1] - 2 * pad
    g = clip_gaze(gaze)
    cx = g[0] * out_w - 0.5 + pad
    cy = g[1] * out_h - 0.5 + pad
    return cv2.getRectSubPix(obs_map, (cs, cs), (float(cx), float(cy)))


def local_features(maps: SaliencyMaps, gaze, cfg_obs: dict) -> np.ndarray:
    """Scalar features: gaze position and window means of the feature maps around the gaze."""
    H, W = maps.shape
    g = clip_gaze(gaze)
    px, py = g[0] * W, g[1] * H
    ww, wh = cfg_obs["feature_window"]
    x0, x1 = int(np.clip(px - ww / 2, 0, W - 1)), int(np.clip(px + ww / 2, 1, W))
    y0, y1 = int(np.clip(py - wh / 2, 0, H - 1)), int(np.clip(py + wh / 2, 1, H))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    win = lambda m: float(m[y0:y1, x0:x1].mean())
    return np.array([g[0], g[1], win(maps.saliency), win(maps.density), win(maps.contrast),
                     win(maps.edges), win(maps.text_mask)], dtype=np.float32)


class ReadingEnv:
    """Gaze-movement simulator over one document image."""

    def __init__(self, image: np.ndarray, cfg: dict, maps: SaliencyMaps | None = None,
                 max_steps: int | None = None):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.cfg_obs = cfg["observation"]
        self.maps = maps if maps is not None else compute_maps(image, cfg["saliency"])
        self.obs_map = observation_map(self.maps.saliency, self.cfg_obs)
        self.max_steps = max_steps if max_steps is not None else cfg["expert"]["max_fixations"]
        self.gaze = None

    def reset(self, gaze) -> dict:
        self.gaze = clip_gaze(gaze)
        self.fixations = [self.gaze.copy()]
        self.dwells: list[float] = []
        self.n_clipped = 0
        return self.observe()

    def observe(self) -> dict:
        if self.gaze is None:
            raise RuntimeError("call reset() first")
        crop = extract_crop(self.obs_map, self.gaze, self.cfg_obs)
        return {"saliency": crop[None].astype(np.float32),
                "features": local_features(self.maps, self.gaze, self.cfg_obs)}

    @property
    def done(self) -> bool:
        return len(self.dwells) >= self.max_steps

    def step(self, displacement, dwell_ms: float):
        """Record the current fixation's dwell, move by ``displacement``, return (obs, done, info)."""
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        d = np.asarray(displacement, dtype=np.float64)
        if d.shape != (2,) or not np.all(np.isfinite(d)):
            raise ValueError(f"displacement must be 2 finite numbers, got {displacement!r}")
        if not np.isfinite(dwell_ms) or dwell_ms <= 0:
            raise ValueError(f"dwell must be positive and finite, got {dwell_ms!r}")
        self.dwells.append(float(dwell_ms))
        raw = self.gaze + d
        self.gaze = clip_gaze(raw)
        clipped = bool(np.any(raw != self.gaze))
        self.n_clipped += int(clipped)
        self.fixations.append(self.gaze.copy())
        return self.observe(), self.done, {"clipped": clipped, "raw_gaze": raw}

    def trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixations that have a recorded dwell, ``(T, 2)``, and their dwells ``(T,)``."""
        n = len(self.dwells)
        return np.array(self.fixations[:n]).reshape(n, 2), np.array(self.dwells)

    def render(self, ax=None, **kwargs):
        from .visualize import draw_scanpath

        fix, dwell = self.trajectory()
        return draw_scanpath(self.image, fix, dwell, ax=ax, **kwargs)
