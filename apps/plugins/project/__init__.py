import os
import json
import shutil
from pathlib import Path
from typing import Optional

from django.conf import settings
from pydantic import validator, DirectoryPath, FilePath

from apps.plugins.frontend import defaults_data
from apps.plugins.frontend.defaults import DefaultsTrainingData
from apps.plugins.project import exceptions
from terra_ai.agent import agent_exchange
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.extra import HardwareAcceleratorData
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.training.extra import (
    LossGraphShowChoice,
    MetricGraphShowChoice,
    ArchitectureChoice,
)
from terra_ai.data.training.train import (
    TrainData,
    TrainingDetailsData,
    LossGraphsList,
    MetricGraphsList,
    ProgressTableList,
    DEFAULT_TRAINING_PATH_NAME,
)
from terra_ai.data.types import confilepath


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

    @validator(
        "base",
        "datasets",
        "modeling",
        "training",
        allow_reuse=True,
        pre=True,
    )
    def _validate_directory(cls, value: DirectoryPath) -> DirectoryPath:
        os.makedirs(value, exist_ok=True)
        return value

    @validator("config", allow_reuse=True, pre=True)
    def _validate_config(cls, value: FilePath) -> FilePath:
        try:
            with open(PROJECT_PATH.get("config"), "x") as _config_ref:
                _config_ref.write("{}")
        except FileExistsError:
            pass
        return value


class Project(BaseMixinData):
    name: str = UNKNOWN_NAME
    dataset: Optional[DatasetData]
    model: ModelDetailsData = ModelDetailsData(**EmptyModelDetailsData)
    training: TrainingDetailsData

    def __init__(self, **data):
        if not data.get("training"):
            data["training"] = {}
        data["training"]["path"] = project_path.training

        if data.get("dataset"):
            data["dataset"]["path"] = project_path.datasets

        super().__init__(**data)

        defaults_data.modeling.set_layer_datatype(self.dataset)
        defaults_data.training = DefaultsTrainingData(
            project=self, architecture=self.training.base.architecture.type
        )

    @property
    def hardware(self) -> HardwareAcceleratorData:
        return agent_exchange("hardware_accelerator")

    @validator("training", pre=True, allow_reuse=True)
    def _validate_training(cls, value, values):
        if not value:
            value = {}
        value.update({"model": values.get("model")})
        return value

    def dict(self, **kwargs):
        _data = super().dict(**kwargs)
        _data.update({"hardware": self.hardware})
        return _data

    def save(self):
        data = self.native()
        if data.get("hardware"):
            data.pop("hardware")
        with open(project_path.config, "w") as _config_ref:
            json.dump(data, _config_ref)

    def frontend(self):
        _data = self.native()
        if _data.get("training", {}).get("deploy") and self.training.deploy:
            _data["training"].update({"deploy": self.training.deploy.presets})
        return json.dumps(_data)

    def set_name(self, name: str):
        self.name = name
        self.save()

    def set_dataset(
        self, dataset: DatasetData, destination: Path, reset_model: bool = False
    ):
        dataset.set_path(destination)
        self.dataset = dataset

        if not self.model.inputs or not self.model.outputs or reset_model:
            self.model = self.dataset.model

        self.model.set_dataset_indexes(self.dataset)
        self.model.update_layers(self.dataset)

        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save()

    def set_model(self, model: ModelDetailsData, clear_dataset: bool = False):
        if clear_dataset:
            self.clear_dataset()
        self.model = model
        if self.dataset:
            self.model.set_dataset_indexes(self.dataset)
            self.model.update_layers(self.dataset)
        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save()

    def set_training(self, name: str = None):
        self.training = TrainingDetailsData(
            name=name, path=project_path.training, model=self.model
        )
        self.set_training_base()
        self.save()

    def set_training_base(self, data: dict = None):
        if data is None:
            data = {}
        self.training.set_base(data, self.dataset)
        defaults_data.training = DefaultsTrainingData(
            project=self, architecture=self.training.base.architecture.type
        )
        self.save()

    def clear_dataset(self):
        self.dataset = None
        shutil.rmtree(project_path.datasets, ignore_errors=True)
        os.makedirs(project_path.datasets, exist_ok=True)
        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save()

    def clear_model(self):
        self.set_model(
            self.dataset.model
            if self.dataset
            else ModelDetailsData(**EmptyModelDetailsData)
        )
        self.save()

    def clear_training(self, name: str):
        shutil.rmtree(Path(project_path.training, name), ignore_errors=True)
        self.set_training(name)
        self.save()

    def _set_data(
        self,
        name: str,
        dataset: DatasetData,
        model: ModelDetailsData,
        training: TrainingDetailsData,
        deploy: DeployData,
    ):
        self.name = name
        self.dataset = dataset
        self.model = model
        self.training = training
        self.deploy = deploy

    def reset(self):
        agent_exchange("training_clear")
        shutil.rmtree(project_path.base, ignore_errors=True)
        ProjectPathData(**PROJECT_PATH)
        self._set_data(
            name=UNKNOWN_NAME,
            dataset=None,
            model=ModelDetailsData(**EmptyModelDetailsData),
            training=TrainingDetailsData(),
            deploy=None,
        )
        self.save()

    def load(self):
        try:
            with open(project_path.config, "r") as _config_ref:
                _config = json.load(_config_ref)
                _dataset = _config.get("dataset", None)
                _model = _config.get("model", None)
                _training = _config.get("training", None)
                _deploy = _config.get("deploy", None)
                self._set_data(
                    name=_config.get("name", UNKNOWN_NAME),
                    dataset=DatasetData(**_dataset) if _dataset else None,
                    model=ModelDetailsData(**(_model or EmptyModelDetailsData)),
                    training=TrainingDetailsData(**(_training or {})),
                    deploy=DeployData(**{"path": project_path.deploy, **_deploy})
                    if _deploy
                    else None,
                )
        except Exception:
            self.reset()

    def update_training_base(self, data: dict = None):
        if isinstance(data, dict):
            if not data.get("architecture"):
                data.update({"architecture": {}})
            data["architecture"].update(
                {
                    "type": self.dataset.architecture.value
                    if self.dataset
                    else ArchitectureChoice.Basic.value,
                    "model": self.model,
                }
            )
            if not data["architecture"].get("parameters"):
                data["architecture"].update({"parameters": {}})
            data["architecture"]["parameters"].update({"model": self.model})
        else:
            data = {
                "architecture": {
                    "type": self.dataset.architecture.value
                    if self.dataset
                    else ArchitectureChoice.Basic.value,
                    "model": self.model,
                    "parameters": {"model": self.model},
                }
            }
        self.training.base = TrainData(**data)
        defaults_data.training = DefaultsTrainingData(
            project=self, architecture=self.training.base.architecture.type
        )
        self.save()


data_path = DataPathData(**DATA_PATH)
project_path = ProjectPathData(**PROJECT_PATH)

try:
    with open(project_path.config, "r") as _config_ref:
        _config = json.load(_config_ref)
except Exception:
    _config = {}

_config.update({"hardware": agent_exchange("hardware_accelerator")})

project = Project(**_config)
project.save()
