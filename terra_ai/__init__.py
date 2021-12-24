import os

from pathlib import Path
from dotenv import load_dotenv


def settings_load(path: str = ".env", **kwargs):
    load_dotenv(str(Path(path).absolute()))
    os.environ.update(**kwargs)
