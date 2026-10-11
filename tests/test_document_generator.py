import numpy as np

from visual_attention.document_generator import generate_document, load_document


def test_image_shape_and_dtype(cfg):
    doc = generate_document(0, cfg["document"], seed=0)
    assert doc.image.shape == (cfg["document"]["height"], cfg["document"]["width"])
    assert doc.image.dtype == np.uint8
    assert doc.image.min() < 100 and doc.image.max() == 255  # has dark text on a white page


def test_layout_is_valid(cfg):
    doc = generate_document(3, cfg["document"], seed=0)
    assert doc.lines[0].is_heading and len(doc.lines) >= 3
    assert len(doc.words) >= 20
    for w in doc.words:
        x0, y0, x1, y1 = w.box
        assert 0 <= x0 < x1 <= doc.width and 0 <= y0 < y1 <= doc.height
        assert 0 < w.contrast <= 1
    # lines are ordered top to bottom and words left to right within a line
    tops = [l.box[1] for l in doc.lines]
    assert tops == sorted(tops)
    for l in doc.lines:
        xs = [doc.words[i].box[0] for i in l.word_ids]
        assert xs == sorted(xs)
    # every word belongs to exactly one line, in reading order
    assert [i for l in doc.lines for i in l.word_ids] == list(range(len(doc.words)))


def test_deterministic_and_varied(cfg):
    a = generate_document(5, cfg["document"], seed=0)
    b = generate_document(5, cfg["document"], seed=0)
    c = generate_document(6, cfg["document"], seed=0)
    assert np.array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)


def test_style_variety(cfg):
    styles, contrasts = set(), []
    for i in range(20):
        d = generate_document(i, cfg["document"], seed=0)
        styles |= {l.style for l in d.lines}
        contrasts += [w.contrast for w in d.words]
    assert {"heading", "dense", "normal", "sparse"} <= styles
    assert min(contrasts) < 0.6 < max(contrasts)  # both low- and high-contrast text


def test_save_load_roundtrip(cfg, tmp_path):
    doc = generate_document(2, cfg["document"], seed=0)
    doc.save(tmp_path)
    back = load_document(tmp_path, 2)
    assert np.array_equal(doc.image, back.image)
    assert [w.box for w in doc.words] == [w.box for w in back.words]
