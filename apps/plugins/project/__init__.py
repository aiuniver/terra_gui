import os
import json
import shutil

from pathlib import Path
from typing import Optional, List, Tuple
from pydantic import validator

from apps.plugins.project import exceptions, utils
from apps.plugins.frontend import defaults_data
from apps.plugins.frontend.defaults import DefaultsTrainingData

from terra_ai import settings as terra_ai_settings
from terra_ai.agent import agent_exchange
from terra_ai.progress import utils as progress_utils
from terra_ai.data.path import ProjectPathData
from terra_ai.data.extra import HardwareAcceleratorData
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.train import TrainingDetailsData, DEFAULT_TRAINING_PATH_NAME
from terra_ai.data.cascades.cascade import CascadeDetailsData
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.presets.training import TasksGroups
from terra_ai.data.presets.cascades import EmptyCascadeDetailsData


UNKNOWN_NAME = "NoName"


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
        data["training"]["path"] = terra_ai_settings.PROJECT_PATH.training

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
    def trainings(self) -> List[Tuple[str, str]]:
        items = []
        for item in os.listdir(terra_ai_settings.PROJECT_PATH.training):
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
        shutil.rmtree(terra_ai_settings.PROJECT_PATH.base, ignore_errors=True)
        terra_ai_settings.PROJECT_PATH = ProjectPathData(
            **terra_ai_settings.PROJECT_PATH.dict()
        )
        self._set_data(
            name=UNKNOWN_NAME,
            dataset_info=None,
            model=ModelDetailsData(**EmptyModelDetailsData),
            cascade=CascadeDetailsData(**EmptyCascadeDetailsData),
        )
        self.set_training()
        self.save_config()
        defaults_data.update_models(self.trainings)

    def save(self, overwrite: bool):
        destination_path = Path(
            terra_ai_settings.TERRA_PATH.projects,
            f"{self.name}.{terra_ai_settings.PROJECT_EXT}",
        )
        if not overwrite and destination_path.is_file():
            raise exceptions.ProjectAlreadyExistsException(self.name)
        zip_destination = progress_utils.pack(
            "project_save",
            "Сохранение проекта",
            terra_ai_settings.PROJECT_PATH.base,
            delete=False,
        )
        shutil.move(zip_destination.absolute(), destination_path.absolute())
        defaults_data.update_models(self.trainings)

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
            with open(terra_ai_settings.PROJECT_PATH.config, "r") as _config_ref:
                _config = json.load(_config_ref)
                _dataset = _config.get("dataset", None)
                if _dataset:
                    _dataset_info = {
                        "alias": _dataset.get("alias"),
                        "group": _dataset.get("group"),
                    }
                else:
                    # _dataset_info = _config.get("dataset_info", None)
                    pass
                _model = _config.get("model", None)
                _cascade = _config.get("cascade", None)
                _training = utils.correct_training(
                    _config.get("training", {}),
                    ModelDetailsData(**(_model or EmptyModelDetailsData)),
                )
                _training["path"] = terra_ai_settings.PROJECT_PATH.training
                # _dataset_info = DatasetInfo(**_dataset_info)
                #     if _dataset_info
                #     else None
                _dataset_info = None
                self._set_data(
                    name=_config.get("name", UNKNOWN_NAME),
                    dataset_info=_dataset_info,
                    model=ModelDetailsData(**(_model or EmptyModelDetailsData)),
                    training=TrainingDetailsData(**_training),
                    cascade=CascadeDetailsData(**(_cascade or EmptyCascadeDetailsData)),
                )
                self.save_config()
                self.set_training(self.training.name)
                defaults_data.update_models(self.trainings)
                terra_ai_settings.PROJECT_PATH = ProjectPathData(
                    **terra_ai_settings.PROJECT_PATH.dict()
                )
        except Exception as error:
            print("ERROR PROJECT LOAD:", error)
            self.create()

    def save_config(self):
        data = self.native()
        if data.get("hardware"):
            data.pop("hardware")
        if data.get("deploy"):
            data.pop("deploy")
        with open(terra_ai_settings.PROJECT_PATH.config, "w") as _config_ref:
            json.dump(data, _config_ref)

    def frontend(self):
        _data = self.native()
        _data.update(
            {
                "deploy": self.deploy.presets if self.deploy else None,
            }
        )
        return json.dumps(_data)

    def set_name(self, name: str):
        self._set_data(name=name)
        self.save_config()

    def set_dataset(self, dataset: DatasetData, reset_model: bool = False):
        self._set_data(dataset=dataset)

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
            self.model.update_layers(self.dataset, exclude_type=True)
        defaults_data.modeling.set_layer_datatype(self.dataset)
        self.clear_training(DEFAULT_TRAINING_PATH_NAME)
        self.save_config()

    def set_training(self, name: str = None):
        self.training = TrainingDetailsData(
            name=name, path=terra_ai_settings.PROJECT_PATH.training, model=self.model
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
        self._set_data(dataset=None)
        shutil.rmtree(terra_ai_settings.PROJECT_PATH.datasets, ignore_errors=True)
        os.makedirs(terra_ai_settings.PROJECT_PATH.datasets, exist_ok=True)
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
        shutil.rmtree(
            Path(terra_ai_settings.PROJECT_PATH.training, name), ignore_errors=True
        )
        self.set_training(name)
        self.save_config()

    def clear_cascade(self):
        self.set_cascade(CascadeDetailsData(**EmptyCascadeDetailsData))


try:
    with open(terra_ai_settings.PROJECT_PATH.config, "r") as _config_ref:
        _config = json.load(_config_ref)
except Exception:
    _config = {}

_config.update({"hardware": agent_exchange("hardware_accelerator")})

project = Project(**_config)
