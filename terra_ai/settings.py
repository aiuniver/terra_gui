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
DATASETS_SOURCE_DIR = Path(TMP_DIR, "datasets", "sources")
DATASETS_LOADED_DIR = Path(TMP_DIR, "datasets", "loaded")

# Modeling
MODEL_EXT = "model"

# Training
TRAINING_DEPLOY_DIRNAME = "deploy"
TRAINING_MODEL_DIRNAME = "model"
TRAINING_INTERMEDIATE_DIRNAME = "intermediate"

# Cascade
CASCADE_EXT = "cascade"
CASCADE_CONFIG = "config.json"
CASCADE_PATH = Path(TMP_DIR, "cascade")

# Deploy
DEPLOY_URL = "https://srv1.demo.neural-university.ru/autodeployterra_upload/"
DEPLOY_PRESET_COUNT = 10
DEPLOY_PRESET_PERCENT = 20
DEPLOY_PATH = Path(TMP_DIR, "deploy")

CALLBACK_CLASSIFICATION_TREASHOLD_VALUE = 90
CALLBACK_REGRESSION_TREASHOLD_VALUE = 2
MAX_GRAPH_LENGTH = 50

# Exceptions
TRANSLATIONS_DIR = Path(ASSETS_PATH, "translations")

# User settings
LANGUAGE = "ru"
