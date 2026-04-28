from pathlib import Path

from src.config import DATA_DIR, DEFAULT_MODEL_PATH, build_config


def test_default_paths_point_to_data_tree():
    config = build_config()

    assert DATA_DIR.name == "data"
    assert "data" in DEFAULT_MODEL_PATH.parts
    assert config.save_root.name == "events"
    assert config.reports_root.name == "reports"
    assert isinstance(config.model_name, Path)
