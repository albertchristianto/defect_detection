import sys
import os
import json
from pathlib import Path

# Add app directory to path for imports
# __file__ is at app/dd/server/api/engine.py
repo_root = Path(__file__).parent.parent.parent.parent  # Go up to repo root
sys.path.insert(0, str(repo_root / "app"))

from dd.backend.py.ImgClassifier import ImgClassifier
from dd.backend.cpp.DdInference import DdInferenceWrapper

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        cfg_path = repo_root / "cfgs" / "AI_System.cfg"
        with open(cfg_path) as f:
            config = json.load(f)

        if config["mode"] == "ImgClassifier":
            if config["backend"] == "py":
                _engine = ImgClassifier(
                    str(repo_root / config["img_classifier_cfg_path"])
                )
            elif config["backend"] == "cpp":
                _engine = DdInferenceWrapper()
    return _engine
