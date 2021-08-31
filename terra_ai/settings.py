"""
## `Конфигурация`
"""

import os

from pathlib import Path
from tempfile import gettempdir


ASSETS_PATH = Path(Path(__file__).parent, "assets")

TMP_DIR = Path(gettempdir(), "terraai")

DATASET_EXT = "trds"
DATASET_CONFIG = "config.json"
DATASET_ANNOTATION = 'labelmap.txt'

MODEL_EXT = "model"

DEPLOY_URL = "https://dev.demo.neural-university.ru/autodeployterra_upload/"

os.makedirs(TMP_DIR, exist_ok=True)
