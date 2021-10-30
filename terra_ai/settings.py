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
DEPLOY_URL = "https://srv1.demo.neural-university.ru/autodeployterra_upload/"
DEPLOY_PRESET_COUNT = 10
DEPLOY_PRESET_PERCENT = 20

CALLBACK_CLASSIFICATION_TREASHOLD_VALUE = 90
CALLBACK_REGRESSION_TREASHOLD_VALUE = 2
MAX_GRAPH_LENGTH = 50

# Exceptions
TRANSLATIONS_DIR = Path(ASSETS_PATH, "translations")

# User settings

LANGUAGE = "ru"
