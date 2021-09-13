import os
import re
import json
import shutil

from typing import Optional
from pathlib import Path
from pydantic import validator, ValidationError, DirectoryPath, FilePath
from transliterate import slugify

from django.conf import settings

from terra_ai.agent import agent_exchange
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilepath
from terra_ai.data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.presets.models import EmptyModelDetailsData

from . import exceptions


UNKNOWN_NAME = "NoName"
DATA_PATH = {
    "base": Path(settings.TERRA_AI_DATA_PATH).absolute(),
    "sources": Path(settings.TERRA_AI_DATA_PATH, "datasets", "sources").absolute(),
    "datasets": Path(settings.TERRA_AI_DATA_PATH, "datasets").absolute(),
    "modeling": Path(settings.TERRA_AI_DATA_PATH, "modeling").absolute(),
    "training": Path(settings.TERRA_AI_DATA_PATH, "training").absolute(),
    "projects": Path(settings.TERRA_AI_DATA_PATH, "projects").absolute(),
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


class ProjectPathData(BaseMixinData):
    base: DirectoryPath
    config: Optional[confilepath(ext="json")]
    datasets: DirectoryPath
    modeling: DirectoryPath
    training: DirectoryPath

    @validator("base", "datasets", "modeling", "training", pre=True)
    def _validate_directory(cls, value: DirectoryPath) -> DirectoryPath:
        os.makedirs(value, exist_ok=True)
        return value

    @validator("config", pre=True)
    def _validate_config(cls, value: FilePath) -> FilePath:
        try:
            with open(PROJECT_PATH.get("config"), "x") as _config_ref:
                _config_ref.write("{}")
        except FileExistsError:
            pass
        return value


class Project(BaseMixinData):
    name: str = UNKNOWN_NAME
    hardware: HardwareAcceleratorData = HardwareAcceleratorData(
        type=HardwareAcceleratorChoice.CPU
    )
    dataset: Optional[DatasetData]
    model: ModelDetailsData = ModelDetailsData(**EmptyModelDetailsData)

    @property
    def name_alias(self) -> str:
        return re.sub(r"([\-]+)", "_", slugify(self.name, language_code="ru"))

    def _set_data(self, name: str, dataset: DatasetData, model: ModelDetailsData):
        self.name = name
        self.dataset = dataset
        self.model = model

    def dict(self, **kwargs):
        _data = super().dict(**kwargs)
        _data.update({"name_alias": self.name_alias})
        return _data

    def reset(self):
        shutil.rmtree(project_path.base, ignore_errors=True)
        ProjectPathData(**PROJECT_PATH)
        self._set_data(
            name=UNKNOWN_NAME,
            dataset=None,
            model=ModelDetailsData(**EmptyModelDetailsData),
        )
        self.save()

    def load(self):
        try:
            with open(project_path.config, "r") as _config_ref:
                _config = json.load(_config_ref)
                _dataset = _config.get("dataset", None)
                _model = _config.get("model", None)
                self._set_data(
                    name=_config.get("name", UNKNOWN_NAME),
                    dataset=DatasetData(**_dataset) if _dataset else None,
                    model=ModelDetailsData(**(_model or EmptyModelDetailsData)),
                )
        except Exception:
            self.reset()

    def save(self):
        with open(project_path.config, "w") as _config_ref:
            json.dump(json.loads(self.json()), _config_ref)

    def set_dataset(self, dataset: DatasetData = None):
        if dataset is None:
            self.dataset = None
            self.model = ModelDetailsData(**EmptyModelDetailsData)
            return
        model_init = dataset.model
        if self.model.inputs and len(self.model.inputs) != len(model_init.inputs):
            raise exceptions.DatasetModelInputsCountNotMatchException()
        if self.model.outputs and len(self.model.outputs) != len(model_init.outputs):
            raise exceptions.DatasetModelOutputsCountNotMatchException()
        self.dataset = dataset
        if not self.model.inputs or not self.model.outputs:
            self.model = model_init
        else:
            layers_init = {"input": [], "output": []}
            for layer in model_init.inputs + model_init.outputs:
                layers_init[layer.group.value].append(layer.native())
            for layer in self.model.inputs + self.model.outputs:
                layer_init = layers_init[layer.group.value].pop(0)
                layer_data = layer.native()
                layer_data.update(
                    {
                        "type": layer_init.get("type"),
                        "shape": layer_init.get("shape"),
                        "task": layer_init.get("task"),
                        "num_classes": layer_init.get("num_classes"),
                        "parameters": layer_init.get("parameters"),
                    }
                )
                self.model.layers.append(layer_data)
                self.model.name = model_init.name
                self.model.alias = model_init.alias

    def set_model(self, model: ModelDetailsData):
        if self.dataset:
            dataset_model = self.dataset.model
            if model.inputs and len(model.inputs) != len(dataset_model.inputs):
                raise exceptions.DatasetModelInputsCountNotMatchException()
            if model.outputs and len(model.outputs) != len(dataset_model.outputs):
                raise exceptions.DatasetModelOutputsCountNotMatchException()
        self.model = model

    def clear_model(self):
        if self.dataset:
            self.model = self.dataset.model
        else:
            self.model = ModelDetailsData(**EmptyModelDetailsData)


data_path = DataPathData(**DATA_PATH)
project_path = ProjectPathData(**PROJECT_PATH)

try:
    with open(project_path.config, "r") as _config_ref:
        _config = json.load(_config_ref)
except Exception:
    _config = {}

_config.update({"hardware": agent_exchange("hardware_accelerator")})
project = Project(**_config)
