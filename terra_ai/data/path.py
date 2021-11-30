import os
import json

from pathlib import Path
from pydantic import validator, BaseModel
from pydantic.types import DirectoryPath


class TerraPathData(BaseModel):
    base: DirectoryPath
    sources: DirectoryPath
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath
    projects: DirectoryPath

    @validator(
        "base",
        "sources",
        "datasets",
        "modeling",
        "training",
        "projects",
        allow_reuse=True,
        pre=True,
    )
    def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
        os.makedirs(value, exist_ok=True)
        return value


class ProjectPathData(BaseModel):
    base: DirectoryPath
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath
    cascades: DirectoryPath
    deploy: DirectoryPath

    @property
    def config(self) -> Path:
        config = Path(self.base, "config.json")
        if not config.is_file():
            with open(config, "w") as config_ref:
                json.dump({}, config_ref)
        return config

    @validator(
        "base",
        "datasets",
        "modeling",
        "training",
        "cascades",
        "deploy",
        allow_reuse=True,
        pre=True,
    )
    def _validate_directory(cls, value: DirectoryPath) -> DirectoryPath:
        os.makedirs(value, exist_ok=True)
        return value
