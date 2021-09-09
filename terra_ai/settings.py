"""
## `Конфигурация`
"""

import os

from pathlib import Path
from tempfile import gettempdir


# General settings
ASSETS_PATH = Path(Path(__file__).parent, "assets")
TMP_DIR = Path(gettempdir(), "terraai")
os.makedirs(TMP_DIR, exist_ok=True)

# Projects
PROJECT_EXT = "project"

# Datasets
DATASET_EXT = "trds"
DATASET_CONFIG = "config.json"
DATASET_ANNOTATION = "labelmap.txt"

# Modeling
MODEL_EXT = "model"

# Deploy
DEPLOY_URL = "https://dev.demo.neural-university.ru/autodeployterra_upload/"
