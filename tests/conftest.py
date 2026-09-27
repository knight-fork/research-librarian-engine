import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, Secrets  # noqa: E402


@pytest.fixture
def cfg():  # radiology profile
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    raw["profile"] = "radiology"  # most tests exercise the built-in radiology ontology; see test_general_profile.py
    return Config(raw=raw, authors=[{"name": "Jane Q. Doe", "auto_add_threshold": 0.72}], secrets=Secrets())


@pytest.fixture
def general_cfg():
    import yaml
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    raw["profile"] = "general"
    return Config(raw=raw, authors=[], secrets=Secrets())
