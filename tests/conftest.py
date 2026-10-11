import pytest

from visual_attention.config import load_config


@pytest.fixture()
def cfg():
    return load_config()


@pytest.fixture()
def small_cfg(tmp_path):
    """Tiny but complete configuration writing everything to a temporary directory."""
    c = load_config()
    c["paths"] = {"data_dir": str(tmp_path / "data"), "checkpoint_dir": str(tmp_path / "ckpt"),
                  "results_dir": str(tmp_path / "results")}
    c["document"]["n_documents"] = 8
    c["expert"]["trajectories_per_document"] = 2
    c["training"]["max_epochs"] = 2
    c["training"]["batch_size"] = 64
    c["training"]["device"] = "cpu"
    c["evaluation"]["scanpath_docs"] = 1
    return c
