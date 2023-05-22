from pathlib import Path
from dataclasses import dataclass


@dataclass
class GlobalSettings:
    TERRA_DIR_PATH: Path
    PROJECT_PATH: Path
