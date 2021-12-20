"""
## `Конфигурация`
"""

import os

from pathlib import Path

from django.conf import settings as django_settings

from .data.path import TerraPathData, ProjectPathData


# General settings
ASSETS_PATH = Path(Path(__file__).parent, "assets")
TMP_DIR = Path(Path(__file__).parent.parent, "Usage")
os.makedirs(TMP_DIR, exist_ok=True)

GOOGLE_STORAGE_URL = "https://storage.googleapis.com/terra_ai/"
WEIGHT_STORAGE_URL = f"{GOOGLE_STORAGE_URL}neural_network/weights/"
WEIGHT_PATH = Path(TMP_DIR, "modeling", "weights")
os.makedirs(WEIGHT_PATH, exist_ok=True)

# Terra paths
TERRA_PATH = TerraPathData(
    **{
        "base": Path(django_settings.TERRA_PATH).absolute(),
        "sources": Path(django_settings.TERRA_PATH, "datasets", "sources").absolute(),
        "datasets": Path(django_settings.TERRA_PATH, "datasets").absolute(),
        "modeling": Path(django_settings.TERRA_PATH, "modeling").absolute(),
        "training": Path(django_settings.TERRA_PATH, "training").absolute(),
        "projects": Path(django_settings.TERRA_PATH, "projects").absolute(),
    }
)

# Project paths
PROJECT_PATH = ProjectPathData(
    **{
        "base": Path(django_settings.PROJECT_PATH).absolute(),
        "datasets": Path(django_settings.PROJECT_PATH, "datasets").absolute(),
        "modeling": Path(django_settings.PROJECT_PATH, "modeling").absolute(),
        "training": Path(django_settings.PROJECT_PATH, "training").absolute(),
        "cascades": Path(django_settings.PROJECT_PATH, "cascades").absolute(),
        "deploy": Path(django_settings.PROJECT_PATH, "deploy").absolute(),
    }
)

# Projects
PROJECT_EXT = "project"

# Datasets
DATASET_EXT = "trds_NEW"  # Окончание _NEW исключительно на момент разработки новой версии.
DATASET_CONFIG = "config.json"
DATASET_ANNOTATION = "labelmap.txt"
DATASETS_SOURCE_DIR = Path(TMP_DIR, "datasets", "sources")
DATASETS_LOADED_DIR = Path(TMP_DIR, "datasets", "loaded")
VERSION_EXT = "vrs"
VERSION_CONFIG = "version.json"

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
