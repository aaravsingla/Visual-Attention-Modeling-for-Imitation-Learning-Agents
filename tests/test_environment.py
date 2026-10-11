import numpy as np
import pytest

from visual_attention.document_generator import generate_document
from visual_attention.environment import FEATURE_NAMES, ReadingEnv, extract_crop, observation_map
from visual_attention.expert import EVENT_NAMES, FORWARD, REGRESSION, RETURN_SWEEP, SKIP, ReadingExpert
from visual_attention.saliency import compute_maps


@pytest.fixture()
def doc_maps(cfg):
    doc = generate_document(0, cfg["document"], seed=0)
    return doc, compute_maps(doc.image, cfg["saliency"])


def test_reset_gives_valid_observation(cfg, doc_maps):
    doc, maps = doc_maps
    env = ReadingEnv(doc.image, cfg, maps)
    obs = env.reset([0.2, 0.1])
    s = cfg["observation"]["crop_size"]
    assert obs["saliency"].shape == (1, s, s) and obs["saliency"].dtype == np.float32
    assert obs["features"].shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(obs["saliency"])) and np.all(np.isfinite(obs["features"]))
    assert obs["saliency"].min() >= 0 and obs["saliency"].max() <= 1
    assert np.allclose(obs["features"][:2], [0.2, 0.1])


def test_step_updates_gaze(cfg, doc_maps):
    doc, maps = doc_maps
    env = ReadingEnv(doc.image, cfg, maps, max_steps=5)
    env.reset([0.3, 0.3])
    obs, done, info = env.step([0.1, -0.05], 200.0)
    assert np.allclose(env.gaze, [0.4, 0.25]) and not info["clipped"] and not done
    assert np.allclose(obs["features"][:2], [0.4, 0.25])


def test_boundary_clipping(cfg, doc_maps):
    doc, maps = doc_maps
    env = ReadingEnv(doc.image, cfg, maps, max_steps=5)
    env.reset([0.95, 0.02])
    _, _, info = env.step([0.2, -0.3], 150.0)
    assert info["clipped"] and np.allclose(env.gaze, [1.0, 0.0])
    obs = env.observe()
    assert np.all(np.isfinite(obs["saliency"]))
    env.reset([-3.0, 7.0])
    assert np.allclose(env.gaze, [0.0, 1.0])


def test_termination_and_invalid_actions(cfg, doc_maps):
    doc, maps = doc_maps
    env = ReadingEnv(doc.image, cfg, maps, max_steps=3)
    env.reset([0.5, 0.5])
    with pytest.raises(ValueError):
        env.step([np.nan, 0.0], 100.0)
    with pytest.raises(ValueError):
        env.step([0.0, 0.0], -5.0)
    dones = [env.step([0.01, 0.0], 100.0)[1] for _ in range(3)]
    assert dones == [False, False, True]
    with pytest.raises(RuntimeError):
        env.step([0.0, 0.0], 100.0)
    fix, dwell = env.trajectory()
    assert fix.shape == (3, 2) and dwell.shape == (3,)


def test_render(cfg, doc_maps, tmp_path):
    import matplotlib.pyplot as plt

    doc, maps = doc_maps
    env = ReadingEnv(doc.image, cfg, maps, max_steps=3)
    env.reset([0.1, 0.1])
    for _ in range(3):
        env.step([0.05, 0.0], 200.0)
    ax = env.render()
    ax.figure.savefig(tmp_path / "r.png")
    plt.close(ax.figure)
    assert (tmp_path / "r.png").stat().st_size > 0


def test_crop_is_centred_on_gaze(cfg, doc_maps):
    doc, maps = doc_maps
    om = observation_map(maps.saliency, cfg["observation"])
    w0 = doc.words[0]
    gaze = [(w0.box[0] + w0.box[2]) / 2 / doc.width, (w0.box[1] + w0.box[3]) / 2 / doc.height]
    crop = extract_crop(om, gaze, cfg["observation"])
    c = crop.shape[0] // 2
    assert crop[c - 2:c + 3, c - 2:c + 3].mean() > crop.mean()  # gaze sits on salient text
    blank = extract_crop(om, [0.5, 0.97], cfg["observation"])  # bottom margin is empty
    assert blank.max() < 0.05


def test_expert_trajectories(cfg, doc_maps):
    doc, maps = doc_maps
    expert = ReadingExpert(doc, maps, cfg["expert"])
    lengths, events = [], []
    for k in range(5):
        tr = expert.sample_trajectory(np.random.default_rng(k))
        T = len(tr["fixations"])
        assert tr["fixations"].shape == (T, 2) and tr["dwell"].shape == (T,) and tr["event"].shape == (T - 1,)
        assert np.all((tr["fixations"] >= 0) & (tr["fixations"] <= 1))
        assert np.all(tr["dwell"] > 0) and np.all(np.isfinite(tr["dwell"]))
        # terminates at the last word or at the fixation budget
        assert tr["word"][-1] == len(doc.words) - 1 or T == cfg["expert"]["max_fixations"]
        assert tr["word"][0] == 0
        lengths.append(T)
        events += tr["event"].tolist()
    events = np.array(events)
    assert len(set(lengths)) > 1  # trajectories differ
    assert np.mean(events == FORWARD) > 0.5  # mostly forward reading
    for ev in (SKIP, REGRESSION, RETURN_SWEEP):
        assert np.any(events == ev), EVENT_NAMES[ev]


def test_expert_reading_order(cfg, doc_maps):
    doc, maps = doc_maps
    expert = ReadingExpert(doc, maps, cfg["expert"])
    tr = expert.sample_trajectory(np.random.default_rng(1))
    w, ev = tr["word"], tr["event"]
    step = np.diff(w)
    assert np.all(step[ev == REGRESSION] < 0)
    assert np.all(step[ev == SKIP] == 2)
    assert np.all(step[ev == FORWARD] == 1)
    assert np.all(step[ev == RETURN_SWEEP] == 1)
    # return sweeps move down to the next line and to the left
    d = np.diff(tr["fixations"], axis=0)
    assert np.all(d[ev == RETURN_SWEEP, 1] > 0) and np.all(d[ev == RETURN_SWEEP, 0] < 0)


def test_expert_max_fixations(cfg, doc_maps):
    doc, maps = doc_maps
    c = dict(cfg["expert"], max_fixations=10)
    tr = ReadingExpert(doc, maps, c).sample_trajectory(np.random.default_rng(0))
    assert len(tr["fixations"]) == 10


def test_dwell_depends_on_features(cfg, doc_maps):
    doc, maps = doc_maps
    expert = ReadingExpert(doc, maps, cfg["expert"])
    d = cfg["expert"]["dwell"]
    i = 0
    expected = (d["base_ms"] + d["per_char_ms"] * len(doc.words[i].text)
                + d["density_ms"] * expert.word_density[i] + d["low_contrast_ms"] * (1 - doc.words[i].contrast))
    assert np.isclose(expert.clean_dwell[i], expected)
    assert expert.clean_dwell.std() > 10  # informative, not constant
