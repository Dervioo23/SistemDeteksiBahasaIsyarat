
import json
import os
from pathlib import Path

# Path ke root project (c:\Deteksi Bahasa Isyarat2)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / "config.json"

class Settings:
    def __init__(self):
        self._config = self._load_config()

    def _load_config(self):
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @property
    def PROJECT_NAME(self):
        return self._config.get("project_name", "Sign Language App")

    @property
    def MODEL_CONFIG(self):
        return self._config.get("model", {})
    
    @property
    def COLLECTION_CONFIG(self):
        return self._config.get("collection", {})
    
    @property
    def INFERENCE_CONFIG(self):
        return self._config.get("inference", {})
    
    @property
    def VOCABULARY(self):
        return self._config.get("vocabulary", {})

    @property
    def UI_CONFIG(self):
        return self._config.get("ui", {})

settings = Settings()
