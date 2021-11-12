import os
import json
import shutil

from pathlib import Path
from typing import Optional
from pydantic import validator, DirectoryPath, FilePath

from django.conf import settings

from apps.plugins.frontend import defaults_data
from apps.plugins.frontend.defaults import DefaultsTrainingData
from apps.plugins.project import exceptions, utils

from terra_ai.settings import PROJECT_EXT
from terra_ai.agent import agent_exchange
from terra_ai.progress import utils as progress_utils
from terra_ai.data.types import confilepath
from terra_ai.data.extra import HardwareAcceleratorData
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.deploy.extra import DeployTypePageChoice
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.train import TrainingDetailsData, DEFAULT_TRAINING_PATH_NAME
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.presets.cascades import EmptyCascadeDetailsData


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
    "cascades": Path(settings.TERRA_AI_PROJECT_PATH, "cascades").absolute(),
    "deploy": Path(settings.TERRA_AI_PROJECT_PATH, "deploy").absolute(),
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
    cascades: DirectoryPath
    deploy: DirectoryPath

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
    cascade: CascadeDetailsData = CascadeDetailsData(**EmptyCascadeDetailsData)
    deploy: Optional[DeployData]

    def __init__(self, **data):
        if not data.get("training"):
            data["training"] = {}
        data["training"]["path"] = project_path.training

        if data.get("dataset"):
            data["dataset"]["path"] = project_path.datasets

        if data.get("deploy"):
            data["deploy"]["path"] = project_path.deploy
            deploy_page = data.get("deploy", {}).get("page", {}).get("type", "")
            if deploy_page == DeployTypePageChoice.model:
                data["deploy"]["path_model"] = project_path.training
            elif deploy_page == DeployTypePageChoice.cascade:
                data["deploy"]["path_model"] = project_path.cascades

        super().__init__(**data)

        defaults_data.modeling.set_layer_datatype(self.dataset)
        defaults_data.training = DefaultsTrainingData(
            project=self, architecture=self.training.base.architecture.type
        )
        defaults_data.update_models(self.trainings)

        self.save_config()

    @property
    def hardware(self) -> HardwareAcceleratorData:
        return agent_exchange("hardware_accelerator")

    @property
    def trainings(self) -> list:
        items = [(DEFAULT_TRAINING_PATH_NAME, "Текущее обучение")]
        for item in os.listdir(project_path.training):
            if item == DEFAULT_TRAINING_PATH_NAME:
                continue
            items.append((item, item))
        return items

    @validator("training", pre=True, allow_reuse=True)
    def _validate_training(cls, value, values):
        if not value:
            value = {}
        value.update({"model": values.get("model")})
        return value

    def _set_data(self, **kwargs):
        kwargs_keys = kwargs.keys()
        if "name" in kwargs_keys:
            self.name = kwargs.get("name")
        if "dataset" in kwargs_keys:
            self.dataset = kwargs.get("dataset")
        if "model" in kwargs_keys:
            self.model = kwargs.get("model")
        if "training" in kwargs_keys:
            self.training = kwargs.get("training")
        if "cascade" in kwargs_keys:
            self.cascade = kwargs.get("cascade")

    def dict(self, **kwargs):
        _data = super().dict(**kwargs)
        _data.update({"hardware": self.hardware})
        return _data

    def create(self):
        # Todo: kill current process of training
        shutil.rmtree(project_path.base, ignore_errors=True)
        ProjectPathData(**PROJECT_PATH)
        self._set_data(
            name=UNKNOWN_NAME,
            dataset=None,
            model=ModelDetailsData(**EmptyModelDetailsData),
            cascade=CascadeDetailsData(**EmptyCascadeDetailsData),
        )
        self.set_training()
        self.save_config()
        defaults_data.update_models(self.trainings)

    def save(self, overwrite: bool):
        destination_path = Path(data_path.projects, f"{self.name}.{PROJECT_EXT}")
        if not overwrite and destination_path.is_file():
            raise exceptions.ProjectAlreadyExistsException(self.name)
        zip_destination = progress_utils.pack(
            "project_save", "Сохранение проекта", project_path.base, delete=False
        )
        shutil.move(zip_destination.name, destination_path)
        defaults_data.update_models(self.trainings)

    def load(self):
        try:
            with open(project_path.config, "r") as _config_ref:
                _config = json.load(_config_ref)
                _dataset = _config.get("dataset", None)
                _model = _config.get("model", None)
                _cascade = _config.get("cascade", None)
                _training = utils.correct_training(
                    _config.get("training", {}),
                    ModelDetailsData(**(_model or EmptyModelDetailsData)),
                )
                if _dataset:
                    _dataset["path"] = project_path.datasets
                _training["path"] = project_path.training
                self._set_data(
                    name=_config.get("name", UNKNOWN_NAME),
                    dataset=DatasetData(**_dataset) if _dataset else None,
                    model=ModelDetailsData(**(_model or EmptyModelDetailsData)),
                    training=TrainingDetailsData(**_training),
                    cascade=CascadeDetailsData(**(_cascade or EmptyCascadeDetailsData)),
                )
                self.save_config()
                self.set_training(self.training.name)
                defaults_data.update_models(self.trainings)
        except Exception as error:
            print("ERROR PROJECT LOAD:", error)
            self.create()

    def save_config(self):
        data = self.native()
        if data.get("hardware"):
            data.pop("hardware")
        with open(project_path.config, "w") as _config_ref:
            json.dump(data, _config_ref)

    def frontend(self):
        _data = self.native()
        if _data.get("training", {}).get("deploy") and self.deploy:
            _data["training"].update({"deploy": self.deploy.presets})
        return json.dumps(_data)

    def set_name(self, name: str):
        self._set_data(name=name)
        self.save_config()

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
        self.save_config()

    def set_model(self, model: ModelDetailsData, clear_dataset: bool = False):
        if clear_dataset:
            self.clear_dataset()
        self.model = model
        if self.dataset:
            self.model.set_dataset_indexes(self.dataset)
            self.model.update_layers(self.dataset)
        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save_config()

    def set_training(self, name: str = None):
        self.training = TrainingDetailsData(
            name=name, path=project_path.training, model=self.model
        )
        self.set_training_base()
        self.save_config()

    def set_training_base(self, data: dict = None):
        if data is None:
            data = {}
        self.training.set_base(data, self.dataset)
        defaults_data.training = DefaultsTrainingData(
            project=self, architecture=self.training.base.architecture.type
        )
        self.training.save(self.training.name)
        self.save_config()

    def set_cascade(self, cascade: CascadeDetailsData):
        self.cascade = cascade
        self.save_config()

    def clear_dataset(self):
        self.dataset = None
        shutil.rmtree(project_path.datasets, ignore_errors=True)
        os.makedirs(project_path.datasets, exist_ok=True)
        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save_config()

    def clear_model(self):
        self.set_model(
            self.dataset.model
            if self.dataset
            else ModelDetailsData(**EmptyModelDetailsData)
        )
        self.save_config()

    def clear_training(self, name: str):
        shutil.rmtree(Path(project_path.training, name), ignore_errors=True)
        self.set_training(name)
        self.save_config()

    def clear_cascade(self):
        self.set_cascade(CascadeDetailsData(**EmptyCascadeDetailsData))


data_path = DataPathData(**DATA_PATH)
project_path = ProjectPathData(**PROJECT_PATH)

try:
    with open(project_path.config, "r") as _config_ref:
        _config = json.load(_config_ref)
except Exception:
    _config = {}

_config.update({"hardware": agent_exchange("hardware_accelerator")})

project = Project(**_config)
