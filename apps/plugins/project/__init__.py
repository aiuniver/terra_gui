import json
import os

from typing import Optional
from pathlib import Path
from pydantic import validator, ValidationError, DirectoryPath

from django.conf import settings

from terra_ai.agent import agent_exchange
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilepath
from terra_ai.data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice


UNKNOWN_NAME = "NoName"
DATA_PATH = {
    "base": Path(settings.TERRA_AI_DATA_PATH).absolute(),
    "sources": Path(settings.TERRA_AI_DATA_PATH, "datasets", "sources").absolute(),
    "datasets": Path(settings.TERRA_AI_DATA_PATH, "datasets").absolute(),
    "modeling": Path(settings.TERRA_AI_DATA_PATH, "modeling").absolute(),
    "training": Path(settings.TERRA_AI_DATA_PATH, "training").absolute(),
}
PROJECT_PATH = {
    "base": Path(settings.TERRA_AI_PROJECT_PATH).absolute(),
    "config": Path(settings.TERRA_AI_PROJECT_PATH, "config.json").absolute(),
    "datasets": Path(settings.TERRA_AI_PROJECT_PATH, "datasets").absolute(),
    "modeling": Path(settings.TERRA_AI_PROJECT_PATH, "modeling").absolute(),
    "training": Path(settings.TERRA_AI_PROJECT_PATH, "training").absolute(),
}


class DataPathData(BaseMixinData):
    base: DirectoryPath
    sources: DirectoryPath
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath

    @validator(
        "base",
        "sources",
        "datasets",
        "modeling",
        "training",
        allow_reuse=True,
        pre=True,
    )
    def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
        try:
            os.makedirs(value)
        except FileExistsError:
            pass
        return value


class ProjectPathData(BaseMixinData):
    base: DirectoryPath
    config: Optional[confilepath(ext="json")]
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath

    @validator("base", "datasets", "modeling", "training", allow_reuse=True, pre=True)
    def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
        try:
            os.makedirs(value)
        except FileExistsError:
            pass
        return value


class DatasetData(BaseMixinData):
    pass


class Project(BaseMixinData):
    name: str = UNKNOWN_NAME
    hardware: HardwareAcceleratorData = HardwareAcceleratorData(
        type=HardwareAcceleratorChoice.CPU
    )
    dataset: Optional[DatasetData]

    def __init__(self, save=False, **data):
        super().__init__(**data)
        if save:
            self.save()

    def save(self):
        with open(project_path.config, "w") as config_ref:
            json.dump(json.loads(self.json()), config_ref)


data_path = DataPathData(**DATA_PATH)

try:
    project_path = ProjectPathData(**PROJECT_PATH)
    project_save = False
except ValidationError as error:
    with open(PROJECT_PATH.get("config"), "x") as config_ref:
        config_ref.write("{}")
    project_path = ProjectPathData(**PROJECT_PATH)
    project_save = True

if project_save:
    project = Project(save=True, hardware=agent_exchange("hardware_accelerator"))
else:
    with open(project_path.config, "r") as config_ref:
        data = json.load(config_ref)
        data.update({"hardware": agent_exchange("hardware_accelerator")})
        project = Project(**data)
