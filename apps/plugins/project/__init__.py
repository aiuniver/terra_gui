import os
import re
import json
import shutil
from pathlib import Path
from typing import Optional

from django.conf import settings
from pydantic import validator, DirectoryPath, FilePath
from transliterate import slugify

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
    InteractiveData,
    StateData,
    LossGraphsList,
    MetricGraphsList,
    ProgressTableList,
)
from terra_ai.data.types import confilepath
from terra_ai.training.guinn import interactive as training_interactive

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
    "deploy": Path(settings.TERRA_AI_PROJECT_PATH, "training", "deploy").absolute(),
    "training_model": Path(
        settings.TERRA_AI_PROJECT_PATH, "training", "model"
    ).absolute(),
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
    deploy: DirectoryPath
    training_model: DirectoryPath

    @validator(
        "base",
        "datasets",
        "modeling",
        "training",
        "deploy",
        "training_model",
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

    def clear_training(self):
        shutil.rmtree(self.training, ignore_errors=True)
        os.makedirs(self.training, exist_ok=True)
        os.makedirs(self.deploy, exist_ok=True)

    def clear_dataset(self):
        shutil.rmtree(self.datasets, ignore_errors=True)
        os.makedirs(self.datasets, exist_ok=True)


class TrainingDetailsData(BaseMixinData):
    name: str = "__current"
    base: TrainData = TrainData()
    interactive: InteractiveData = InteractiveData()
    state: StateData = StateData()
    result: Optional[dict]

    def set_state(self):
        self.state = StateData(**training_interactive.train_states)


class Project(BaseMixinData):
    name: str = UNKNOWN_NAME
    dataset: Optional[DatasetData]
    model: ModelDetailsData = ModelDetailsData(**EmptyModelDetailsData)
    training: TrainingDetailsData = TrainingDetailsData()
    deploy: Optional[DeployData]

    @property
    def name_alias(self) -> str:
        return re.sub(r"([\-]+)", "_", slugify(self.name, language_code="ru"))

    @property
    def hardware(self) -> HardwareAcceleratorData:
        return agent_exchange("hardware_accelerator")

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

    def dict(self, **kwargs):
        _data = super().dict(**kwargs)
        _data.update(
            {
                "name_alias": self.name_alias,
                "hardware": self.hardware,
            }
        )
        return _data

    def front(self):
        _data = self.native()
        if _data.get("deploy") and self.deploy:
            _data.update({"deploy": self.deploy.presets})
        return json.dumps(_data)

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

    def save(self):
        with open(project_path.config, "w") as _config_ref:
            json.dump(json.loads(self.json()), _config_ref)

    def clear_training(self):
        self.deploy = None
        project_path.clear_training()
        self.training = TrainingDetailsData()
        self.save()

    def _redefine_model_ids(self):
        if not self.dataset:
            return
        dataset_model = self.dataset.model
        for _index, _dataset_layer in enumerate(dataset_model.inputs):
            self.model.switch_index(self.model.inputs[_index].id, _dataset_layer.id)
        for _index, _dataset_layer in enumerate(dataset_model.outputs):
            self.model.switch_index(self.model.outputs[_index].id, _dataset_layer.id)

    def update_model_layers(self):
        if not self.dataset:
            return

        model_init = self.dataset.model

        for index, layer in enumerate(self.model.inputs):
            layer_init = model_init.inputs.get(layer.id)
            layer.shape = layer_init.shape
            layer.task = layer_init.task
            layer.num_classes = layer_init.num_classes
            # layer.parameters = layer_init.parameters

        for index, layer in enumerate(self.model.outputs):
            layer_init = model_init.outputs.get(layer.id)
            layer.shape = layer_init.shape
            layer.task = layer_init.task
            layer.num_classes = layer_init.num_classes
            # layer.parameters = layer_init.parameters

    def set_dataset(self, dataset: DatasetData = None, reset_model: bool = False):
        if dataset is None:
            self.dataset = None
            project_path.clear_dataset()
            defaults_data.modeling.set_layer_datatype(self.dataset)
            self.set_training()
            return

        self.dataset = dataset
        if not self.model.inputs or not self.model.outputs or reset_model:
            self.model = self.dataset.model
        else:
            self._redefine_model_ids()
            self.update_model_layers()

        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.set_training()
        self.save()

    def set_model(self, model: ModelDetailsData):
        if self.dataset:
            dataset_model = self.dataset.model
            if model.inputs and len(model.inputs) != len(dataset_model.inputs):
                raise exceptions.DatasetModelInputsCountNotMatchException()
            if model.outputs and len(model.outputs) != len(dataset_model.outputs):
                raise exceptions.DatasetModelOutputsCountNotMatchException()
        self.model = model
        self._redefine_model_ids()
        self.update_model_layers()
        self.set_training()
        self.save()

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

    def update_training_interactive(self):
        loss_graphs = []
        metric_graphs = []
        progress_table = []
        _index_m = 0
        _index_l = 0
        for layer in self.model.outputs:
            outputs = self.training.base.architecture.parameters.outputs.get(layer.id)
            if not outputs:
                continue
            for metric in outputs.metrics:
                _index_m += 1
                metric_graphs.append(
                    {
                        "id": _index_m,
                        "output_idx": layer.id,
                        "show": MetricGraphShowChoice.model,
                        "show_metric": metric,
                    }
                )
                _index_m += 1
                metric_graphs.append(
                    {
                        "id": _index_m,
                        "output_idx": layer.id,
                        "show": MetricGraphShowChoice.classes,
                        "show_metric": metric,
                    }
                )
            _index_l += 1
            loss_graphs.append(
                {
                    "id": _index_l,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.model,
                }
            )
            _index_l += 1
            loss_graphs.append(
                {
                    "id": _index_l,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.classes,
                }
            )
            progress_table.append(
                {
                    "output_idx": layer.id,
                }
            )
        self.training.interactive.loss_graphs = LossGraphsList(loss_graphs)
        self.training.interactive.metric_graphs = MetricGraphsList(metric_graphs)
        self.training.interactive.progress_table = ProgressTableList(progress_table)
        self.training.interactive.intermediate_result.main_output = (
            self.model.outputs[0].id if len(self.model.outputs) else None
        )

    def update_training_state(self):
        self.training.set_state()

    def set_training(self, data: dict = None):
        self.update_training_base(data.get("base") if data else None)
        self.update_training_interactive()
        self.update_training_state()

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
if _config.get("deploy"):
    _config["deploy"].update({"path": project_path.deploy})
_training = _config.pop("training") if _config.get("training") else {}
project = Project(**_config)
project.set_training(_training)
defaults_data.modeling.set_layer_datatype(project.dataset)
