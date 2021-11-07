import os
import json
import shutil

from pathlib import Path
from typing import Optional
from pydantic import validator, DirectoryPath, FilePath

from django.conf import settings

from apps.plugins.frontend import defaults_data
from apps.plugins.frontend.defaults import DefaultsTrainingData
from apps.plugins.project import exceptions

from terra_ai.agent import agent_exchange
from terra_ai.data.types import confilepath
from terra_ai.data.extra import HardwareAcceleratorData
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.train import TrainingDetailsData, DEFAULT_TRAINING_PATH_NAME
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.presets.cascades import EmptyCascadeDetailsData
from terra_ai.data.presets.training import TasksGroups


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

    @validator(
        "base",
        "datasets",
        "modeling",
        "training",
        "cascades",
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

    def clear_cascade(self):
        self.cascade = CascadeDetailsData(**EmptyCascadeDetailsData)

    def set_cascade(self, cascade: CascadeDetailsData):
        self.cascade = cascade
        self.save()

    def _set_data(
        self,
        name: str,
        dataset: DatasetData,
        model: ModelDetailsData,
        training: TrainingDetailsData,
        cascade: CascadeDetailsData,
        deploy: DeployData = None,
    ):
        self.name = name
        self.dataset = dataset
        self.model = model
        self.training = training
        self.cascade = cascade
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
            cascade=CascadeDetailsData(**EmptyCascadeDetailsData),
            deploy=None,
        )
        self.save()

    def load(self):
        def _correct_training(data: dict, model: ModelDetailsData):
            if not data.get("base"):
                data["base"] = {}
            if not data["base"].get("architecture"):
                data["base"]["architecture"] = {}
            data["base"]["architecture"].update({"model": model})
            if not data["base"]["architecture"].get("parameters"):
                data["base"]["architecture"]["parameters"] = {}
            if not data["base"]["architecture"]["parameters"].get("outputs"):
                data["base"]["architecture"]["parameters"]["outputs"] = []
            _outputs = (
                data.get("base", {})
                .get("architecture", {})
                .get("parameters", {})
                .get("outputs", [])
            )
            _outputs_correct = []
            for _output in _outputs:
                _metrics = _output.get("metrics", [])
                _loss = _output.get("loss", "")
                _task = _output.get("task")
                if not _task:
                    _metrics = []
                    _loss = ""
                else:
                    _task_groups = list(
                        filter(lambda item: item.get("task") == _task, TasksGroups)
                    )
                    _task_group = _task_groups[0] if len(_task_groups) else None
                    if _task_group:
                        _metrics = list(set(_metrics) & set(_task_group.get("metrics")))
                        if not len(_metrics):
                            _metrics = [_task_group.get("metrics")[0].value]
                        if _loss not in _task_group.get("losses"):
                            _loss = _task_group.get("losses")[0].value
                    else:
                        _metrics = []
                        _loss = ""
                _output["metrics"] = _metrics
                _output["loss"] = _loss
                _outputs_correct.append(_output)
            data["base"]["architecture"]["parameters"]["outputs"] = _outputs_correct
            _checkpoint = _outputs = (
                data.get("base", {})
                .get("architecture", {})
                .get("parameters", {})
                .get("checkpoint", {})
            )
            if _checkpoint:
                _layer = _checkpoint.get("layer")
                _metric_name = _checkpoint.get("metric_name")
                _outputs = list(
                    filter(lambda item: item.get("id") == _layer, _outputs_correct)
                )
                _output = _outputs[0] if len(_outputs) else None
                if _output:
                    if _metric_name not in _output.get("metrics"):
                        _metric_name = (
                            _output.get("metrics")[0]
                            if len(_output.get("metrics"))
                            else ""
                        )
                else:
                    _layer = ""
                    _metric_name = ""
                _checkpoint["layer"] = _layer
                _checkpoint["metric_name"] = _metric_name
                data["base"]["architecture"]["parameters"]["checkpoint"] = _checkpoint
            data["interactive"] = {}
            return data

        try:
            with open(project_path.config, "r") as _config_ref:
                _config = json.load(_config_ref)
                _dataset = _config.get("dataset", None)
                _model = _config.get("model", None)
                _training = _config.get("training", None)
                _cascade = _config.get("cascade", None)
                self._set_data(
                    name=_config.get("name", UNKNOWN_NAME),
                    dataset=DatasetData(**_dataset) if _dataset else None,
                    model=ModelDetailsData(**(_model or EmptyModelDetailsData)),
                    cascade=CascadeDetailsData(**(_cascade or EmptyCascadeDetailsData)),
                    training=TrainingDetailsData(
                        **(
                            _correct_training(
                                _training or {},
                                ModelDetailsData(**(_model or EmptyModelDetailsData)),
                            )
                        )
                    ),
                )
                self.set_training(_training)
                self.save()
        except Exception as error:
            print("ERROR PROJECT LOAD:", error)
            self.reset()


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
