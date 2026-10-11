import numpy as np

from visual_attention.document_generator import generate_document
from visual_attention.saliency import compute_maps, downsample, overlay

MAP_NAMES = ("saliency", "density", "text_mask", "contrast", "edges", "ink")


def test_shapes_finite_and_normalised(cfg):
    doc = generate_document(0, cfg["document"], seed=0)
    maps = compute_maps(doc.image, cfg["saliency"])
    for name in MAP_NAMES:
        m = getattr(maps, name)
        assert m.shape == doc.image.shape and m.dtype == np.float32
        assert np.all(np.isfinite(m)) and m.min() >= 0 and m.max() <= 1
    assert maps.saliency.max() == 1.0  # min-max normalised


def test_blank_and_constant_images(cfg):
    for value in (255, 0, 128):
        img = np.full((800, 600), value, dtype=np.uint8)
        maps = compute_maps(img, cfg["saliency"])
        for name in MAP_NAMES:
            m = getattr(maps, name)
            assert np.all(np.isfinite(m)) and np.all(m == 0), (value, name)


def test_nearly_constant_image(cfg):
    rng = np.random.default_rng(0)
    img = (200 + rng.integers(0, 2, size=(100, 80))).astype(np.uint8)
    maps = compute_maps(img, cfg["saliency"])
    assert np.all(np.isfinite(maps.saliency)) and maps.saliency.max() <= 1


def test_deterministic(cfg):
    doc = generate_document(1, cfg["document"], seed=0)
    a = compute_maps(doc.image, cfg["saliency"])
    b = compute_maps(doc.image.copy(), cfg["saliency"])
    for name in MAP_NAMES:
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_text_is_salient_and_dense(cfg):
    doc = generate_document(0, cfg["document"], seed=0)
    maps = compute_maps(doc.image, cfg["saliency"])
    inside = np.zeros(doc.image.shape, bool)
    for w in doc.words:
        x0, y0, x1, y1 = w.box
        inside[y0:y1, x0:x1] = True
    assert maps.saliency[inside].mean() > 3 * maps.saliency[~inside].mean()
    assert maps.density[inside].mean() > 5 * maps.density[~inside].mean()
    # Tight word boxes include sparse ascender/descender rows, so test the mask on the central
    # band of every word, and check that it rarely fires away from the text lines.
    centre = np.zeros(doc.image.shape, bool)
    near_lines = np.zeros(doc.image.shape, bool)
    for w in doc.words:
        x0, y0, x1, y1 = w.box
        q = (y1 - y0) // 4
        centre[y0 + q:y1 - q, x0:x1] = True
    for l in doc.lines:
        x0, y0, x1, y1 = l.box
        near_lines[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4] = True
    assert maps.text_mask[centre].mean() > 0.95
    assert maps.text_mask[~near_lines].mean() < 0.01


def test_saliency_and_density_are_distinct(cfg):
    doc = generate_document(0, cfg["document"], seed=0)
    maps = compute_maps(doc.image, cfg["saliency"])
    assert not np.allclose(maps.saliency, maps.density)


def test_downsample_and_overlay(cfg, tmp_path):
    import matplotlib.pyplot as plt

    doc = generate_document(0, cfg["document"], seed=0)
    maps = compute_maps(doc.image, cfg["saliency"])
    small = downsample(maps.saliency, 84)
    assert small.shape == (84, 84) and np.all(np.isfinite(small))
    ax, _ = overlay(doc.image, small)
    ax.figure.savefig(tmp_path / "overlay.png")
    plt.close(ax.figure)
    assert (tmp_path / "overlay.png").stat().st_size > 0
