import os
import re
import json
import shutil

from typing import Optional, Dict
from pathlib import Path
from pydantic import validator, DirectoryPath, FilePath
from transliterate import slugify

from django.conf import settings

from apps.plugins.project import exceptions
from apps.plugins.frontend import defaults_data
from apps.plugins.frontend.presets.defaults import TrainingTasksRelations

from terra_ai import settings as terra_settings
from terra_ai.agent import agent_exchange
from terra_ai.training.guinn import interactive as training_interactive
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import confilepath
from terra_ai.data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.modeling.layer import LayerData
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.training.train import (
    TrainData,
    InteractiveData,
    StateData,
    LossGraphsList,
    MetricGraphsList,
    ProgressTableList,
)
from terra_ai.data.training.outputs import OutputsList
from terra_ai.data.training.checkpoint import CheckpointData
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice
from terra_ai.data.deploy import tasks as deploy_tasks
from terra_ai.data.deploy.extra import TaskTypeChoice as DeployTaskTypeChoice


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
}

TASKS_RELATIONS = {
    DeployTaskTypeChoice.image_classification: {"ImageClassification"},
    DeployTaskTypeChoice.image_segmentation: {"ImageSegmentation"},
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

    @validator(
        "base", "datasets", "modeling", "training", "deploy", allow_reuse=True, pre=True
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


class TrainingDetailsData(BaseMixinData):
    base: TrainData = TrainData()
    interactive: InteractiveData = InteractiveData()
    state: StateData = StateData()
    result: Optional[dict]

    def set_state(self):
        self.state = StateData(**training_interactive.train_states)


class DeployDetailsData(BaseMixinData):
    type: Optional[DeployTaskTypeChoice]
    data: Dict[str, deploy_tasks.BaseCollection] = {}


class Project(BaseMixinData):
    name: str = UNKNOWN_NAME
    hardware: HardwareAcceleratorData = HardwareAcceleratorData(
        type=HardwareAcceleratorChoice.CPU
    )
    dataset: Optional[DatasetData]
    model: ModelDetailsData = ModelDetailsData(**EmptyModelDetailsData)
    training: TrainingDetailsData = TrainingDetailsData()
    deploy: DeployDetailsData = DeployDetailsData()

    @property
    def name_alias(self) -> str:
        return re.sub(r"([\-]+)", "_", slugify(self.name, language_code="ru"))

    def _set_data(
        self,
        name: str,
        dataset: DatasetData,
        model: ModelDetailsData,
        training: TrainingDetailsData,
        deploy: DeployDetailsData,
    ):
        self.name = name
        self.dataset = dataset
        self.model = model
        self.training = training
        self.deploy = deploy

    def dict(self, **kwargs):
        _data = super().dict(**kwargs)
        _data.update({"name_alias": self.name_alias})
        return _data

    def reset(self):
        agent_exchange("training_clear")
        shutil.rmtree(project_path.base, ignore_errors=True)
        ProjectPathData(**PROJECT_PATH)
        self._set_data(
            name=UNKNOWN_NAME,
            dataset=None,
            model=ModelDetailsData(**EmptyModelDetailsData),
            training=TrainingDetailsData(),
            deploy=DeployDetailsData(),
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
                    deploy=DeployDetailsData(**(_deploy or {})),
                )
        except Exception:
            self.reset()

    def save(self):
        with open(project_path.config, "w") as _config_ref:
            json.dump(json.loads(self.json()), _config_ref)

    def set_deploy(self):
        model = self.dataset.model if self.dataset else None
        tasks = {}
        if model:
            for _input in model.inputs:
                for _output in model.outputs:
                    tasks.update(
                        {
                            f"{_input.group}{_input.id}_{_output.group}{_output.id}": f"{_input.task.name}{_output.task.name}"
                        }
                    )
        try:
            _type = list(TASKS_RELATIONS.keys())[
                list(TASKS_RELATIONS.values()).index(set(tasks.values()))
            ]
        except Exception:
            self.deploy = DeployDetailsData()
            return

        data = {}
        for _id, _task in tasks.items():
            _task_class = getattr(deploy_tasks, f"{_task}CollectionList", None)
            if not _task_class:
                continue
            data.update(
                {
                    _id: {
                        "type": _task,
                        "data": _task_class(
                            list(
                                map(
                                    lambda _: None,
                                    list(range(terra_settings.DEPLOY_PRESET_COUNT)),
                                )
                            ),
                            path=Path(project_path.deploy),
                        ),
                    }
                }
            )
        self.deploy = DeployDetailsData(**{"type": _type, "data": data})
        for item in self.deploy.data.values():
            item.data.reload(list(range(terra_settings.DEPLOY_PRESET_COUNT)))
            item.data.try_init()

    def set_dataset(self, dataset: DatasetData = None):
        if dataset is None:
            self.dataset = None
            self.model = ModelDetailsData(**EmptyModelDetailsData)
            self.set_training()
            self.set_deploy()
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
            for index, layer in enumerate(model_init.inputs):
                layer_init = layer.native()
                layer_data = self.model.inputs[index].native()
                layer_data.update(
                    {
                        "type": layer_init.get("type"),
                        "shape": layer_init.get("shape"),
                        "task": layer_init.get("task"),
                        "num_classes": layer_init.get("num_classes"),
                        "parameters": layer_init.get("parameters"),
                    }
                )
                self.model.layers.append(LayerData(**layer_data))
            for index, layer in enumerate(model_init.outputs):
                layer_init = layer.native()
                layer_data = self.model.outputs[index].native()
                layer_data.update(
                    {
                        "type": layer_init.get("type"),
                        "shape": layer_init.get("shape"),
                        "task": layer_init.get("task"),
                        "num_classes": layer_init.get("num_classes"),
                        "parameters": layer_init.get("parameters"),
                    }
                )
                self.model.layers.append(LayerData(**layer_data))
            self.model.name = model_init.name
            self.model.alias = model_init.alias
        self.set_training()
        self.set_deploy()

    def set_model(self, model: ModelDetailsData):
        if self.dataset:
            dataset_model = self.dataset.model
            if model.inputs and len(model.inputs) != len(dataset_model.inputs):
                raise exceptions.DatasetModelInputsCountNotMatchException()
            if model.outputs and len(model.outputs) != len(dataset_model.outputs):
                raise exceptions.DatasetModelOutputsCountNotMatchException()
        self.model = model
        self.set_training()

    def update_training_base(self):
        outputs = []
        for layer in self.model.outputs:
            training_layer = self.training.base.architecture.parameters.outputs.get(
                layer.id
            )
            training_task_rel = TrainingTasksRelations.get(layer.task)
            available_metrics = list(
                set(training_layer.metrics if training_layer else [])
                & set(training_task_rel.metrics if training_task_rel else [])
            )
            training_losses = (
                list(map(lambda item: item.name, training_task_rel.losses))
                if training_task_rel
                else None
            )
            training_metrics = (
                list(map(lambda item: item.name, training_task_rel.metrics))
                if training_task_rel
                else None
            )
            outputs.append(
                {
                    "id": layer.id,
                    "classes_quantity": layer.num_classes,
                    "task": layer.task,
                    "loss": training_layer.loss
                    if training_layer
                    else (training_losses[0] if training_losses else None),
                    "metrics": available_metrics
                    if available_metrics
                    else ([training_metrics[0]] if training_metrics else []),
                }
            )
        self.training.base.architecture.parameters.outputs = OutputsList(outputs)
        if self.model.outputs:
            checkpoint_data = {"layer": self.model.outputs[0].id}
            if self.training.base.architecture.parameters.checkpoint:
                checkpoint_data = (
                    self.training.base.architecture.parameters.checkpoint.native()
                )
                if not checkpoint_data.get("layer"):
                    checkpoint_data.update({"layer": self.model.outputs[0].id})
            self.training.base.architecture.parameters.checkpoint = CheckpointData(
                **checkpoint_data
            )
        defaults_data.update_by_model(self.model, self.training)

    def update_training_interactive(self):
        loss_graphs = []
        metric_graphs = []
        progress_table = []
        index = 0
        for layer in self.model.outputs:
            index += 1
            layer_for_metrics = self.training.base.architecture.parameters.outputs.get(
                layer.id
            )
            metrics = layer_for_metrics.metrics if layer_for_metrics else None
            loss_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.model,
                }
            )
            metric_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": MetricGraphShowChoice.model,
                    "show_metric": metrics[0] if metrics else None,
                }
            )
            index += 1
            loss_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": LossGraphShowChoice.classes,
                }
            )
            metric_graphs.append(
                {
                    "id": index,
                    "output_idx": layer.id,
                    "show": MetricGraphShowChoice.classes,
                    "show_metric": metrics[0] if metrics else None,
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

    def set_training(self):
        self.update_training_base()
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
project = Project(**_config)
project.set_training()
project.set_deploy()
