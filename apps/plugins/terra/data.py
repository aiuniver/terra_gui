import os
import json
import shutil

import yaml
import cairosvg
import pydantic
import tensorflow

from enum import Enum
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

from django.conf import settings
from django.urls import reverse_lazy

from . import utils as terra_utils


class Color(str, Enum):
    red = "\033[0;31m"
    reset = "\033[0m"


class Hardware(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    TRU = "TRU"


class OptimizerType(str, Enum):
    SGD = "SGD"
    RMSprop = "RMSprop"
    Adam = "Adam"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"


class TaskType(str, Enum):
    classification = "classification"
    timeseries = "timeseries"
    regression = "regression"
    segmentation = "segmentation"


class CheckpointIndicatorType(str, Enum):
    train = "train"
    val = "val"


class CheckpointModeType(str, Enum):
    min = "min"
    max = "max"


class LayerLocation(str, Enum):
    input = "input"
    middle = "middle"
    output = "output"


class LayerType(str, Enum):
    Input = "Input"
    Conv1D = "Conv1D"
    Conv2D = "Conv2D"
    Conv3D = "Conv3D"
    Conv1DTranspose = "Conv1DTranspose"
    Conv2DTranspose = "Conv2DTranspose"
    SeparableConv1D = "SeparableConv1D"
    SeparableConv2D = "SeparableConv2D"
    DepthwiseConv2D = "DepthwiseConv2D"
    MaxPooling1D = "MaxPooling1D"
    MaxPooling2D = "MaxPooling2D"
    AveragePooling1D = "AveragePooling1D"
    AveragePooling2D = "AveragePooling2D"
    UpSampling1D = "UpSampling1D"
    UpSampling2D = "UpSampling2D"
    LeakyReLU = "LeakyReLU"
    Dropout = "Dropout"
    Dense = "Dense"
    Add = "Add"
    Multiply = "Multiply"
    Flatten = "Flatten"
    Concatenate = "Concatenate"
    Reshape = "Reshape"
    sigmoid = "sigmoid"
    softmax = "softmax"
    tanh = "tanh"
    relu = "relu"
    elu = "elu"
    selu = "selu"
    PReLU = "PReLU"
    GlobalMaxPooling1D = "GlobalMaxPooling1D"
    GlobalMaxPooling2D = "GlobalMaxPooling2D"
    GlobalAveragePooling1D = "GlobalAveragePooling1D"
    GlobalAveragePooling2D = "GlobalAveragePooling2D"
    GRU = "GRU"
    LSTM = "LSTM"
    Embedding = "Embedding"
    RepeatVector = "RepeatVector"
    BatchNormalization = "BatchNormalization"


class OptimizerParams(pydantic.BaseModel):
    main: Dict[str, Optional[Any]] = {}
    extra: Dict[str, Optional[Any]] = {}

    @pydantic.validator("main", "extra", allow_reuse=True)
    def correct_dict_str_values(cls, value):
        for name, item in value.items():
            try:
                if item is None:
                    item = ""
                if isinstance(item, (tuple, list)):
                    item = ",".join(list(map(lambda value: str(value), item)))
            except Exception:
                item = ""
            value[name] = item
        return value


class Optimizer(pydantic.BaseModel):
    name: OptimizerType = OptimizerType.Adam
    params: OptimizerParams = OptimizerParams()


class Checkpoint(pydantic.BaseModel):
    indicator: CheckpointIndicatorType = CheckpointIndicatorType.val
    monitor: Dict[str, str] = {
        "output": "output_1",
        "out_type": "metrics",
        "out_monitor": "accuracy",
    }  # need to reformat
    mode: CheckpointModeType = CheckpointModeType.max
    save_best: bool = True
    save_weights: bool = False


class ModelPlan(pydantic.BaseModel):
    framework: str = "keras"
    input_datatype: str = "2D"
    plan_name: str = ""
    num_classes: int = 10
    input_shape: Dict[str, Optional[Any]] = {"input_1": (28, 28, 1)}
    output_shape: Dict[str, Optional[Any]] = {"output_1": (28, 28, 1)}
    plan: List[tuple] = []


class OutputConfig(pydantic.BaseModel):
    task: TaskType = TaskType.classification
    loss: str = ""
    metrics: List[str] = []
    num_classes: int = 0
    callbacks: Dict[str, bool] = {}


class TrainConfig(pydantic.BaseModel):
    batch_sizes: int = 32
    epochs_count: int = 20
    optimizer: Optimizer = Optimizer()
    outputs: Dict[str, OutputConfig] = {}
    checkpoint: Checkpoint = Checkpoint()


class LayerConfigParam(pydantic.BaseModel):
    main: Dict[str, Optional[Any]] = {}
    extra: Dict[str, Optional[Any]] = {}

    @pydantic.validator("main", "extra", allow_reuse=True)
    def correct_dict_str_values(cls, value):
        for name, item in value.items():
            try:
                if item is None:
                    item = ""
                if isinstance(item, (tuple, list)):
                    item = ",".join(list(map(lambda value: str(value), item)))
            except Exception:
                item = ""
            value[name] = item
        return value


class LayerConfig(pydantic.BaseModel):
    name: str = ""
    dts_layer_name: str = ""
    type: LayerType = LayerType.Dense
    location_type: LayerLocation = LayerLocation.middle
    up_link: List[int] = []
    input_shape: Union[List[int], List[List[int]]] = []
    output_shape: List[int] = []
    data_name: str = ""
    data_available: List[str] = []
    params: LayerConfigParam = LayerConfigParam()
    num_classes: int = 0

    @pydantic.validator("up_link", allow_reuse=True)
    def correct_list_natural_number(cls, value):
        value = list(filter(lambda value: int(value) > 0, value))
        value = list(set(value))
        return value


class Layer(pydantic.BaseModel):
    x: Optional[float]
    y: Optional[float]
    down_link: List[int] = []
    config: LayerConfig = LayerConfig()

    @pydantic.validator("down_link", allow_reuse=True)
    def validate_list_natural_number(cls, value):
        value = list(filter(lambda value: value > 0, value))
        value = list(set(value))
        return value


class Dataset(pydantic.BaseModel):
    name: str = ""
    tags: dict = {}
    date: Optional[str] = None
    size: Optional[str] = None


class GoogleDrivePath(pydantic.BaseModel):
    datasets: str = f"{settings.TERRA_AI_DATA_PATH}/datasets"
    datasets_sources: str = f"{settings.TERRA_AI_DATA_PATH}/datasets/sources"
    modeling: str = f"{settings.TERRA_AI_DATA_PATH}/modeling"
    training: str = f"{settings.TERRA_AI_DATA_PATH}/training"
    projects: str = f"{settings.TERRA_AI_DATA_PATH}/projects"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.datasets, exist_ok=True)
        os.makedirs(self.datasets_sources, exist_ok=True)
        os.makedirs(self.modeling, exist_ok=True)
        os.makedirs(self.training, exist_ok=True)
        os.makedirs(self.projects, exist_ok=True)


class ProjectPath(pydantic.BaseModel):
    datasets: str = f"{settings.TERRA_AI_PROJECT_PATH}/datasets"
    modeling: str = f"{settings.TERRA_AI_PROJECT_PATH}/modeling"
    training: str = f"{settings.TERRA_AI_PROJECT_PATH}/training"
    config: str = f"{settings.TERRA_AI_PROJECT_PATH}/project.conf"

    _modeling_plan = "plan.yaml"
    _modeling_preview = "preview.png"
    _modeling_keras = "keras.py"
    _modeling_layers = "layers.json"
    _training_output = "output.json"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.datasets, exist_ok=True)
        os.makedirs(self.modeling, exist_ok=True)
        os.makedirs(self.training, exist_ok=True)

    @property
    def validated(self) -> bool:
        return os.path.isfile(
            os.path.join(self.modeling, self._modeling_plan)
        ) and os.path.isfile(os.path.join(self.modeling, self._modeling_keras))

    @property
    def keras_code(self) -> (bool, str):
        success = False
        output = ""
        try:
            with open(
                os.path.join(self.modeling, self._modeling_keras), "r"
            ) as keras_file:
                success = True
                output = keras_file.read()
        except Exception as error:
            output = str(error)
        return success, output

    @property
    def training_output(self) -> dict:
        filepath = os.path.join(self.training, self._training_output)
        if os.path.isfile(filepath):
            with open(filepath, "r") as training_file:
                return json.load(training_file)
        else:
            return {}

    def save_training_output(self, training: dict):
        filepath = os.path.join(self.training, self._training_output)
        with open(filepath, "w") as training_file:
            json.dump(training, training_file)

    def create_plan(self, plan: dict):
        filepath = os.path.join(self.modeling, self._modeling_plan)
        with open(filepath, "w") as yaml_file:
            yaml.dump(plan, yaml_file)

    def create_preview(self, preview: str):
        filepath = os.path.join(self.modeling, self._modeling_preview)
        cairosvg.svg2png(preview, write_to=filepath)
        terra_utils.autocrop_image_square(filepath)

    def create_keras(self, keras: str):
        filepath = os.path.join(self.modeling, self._modeling_keras)
        with open(filepath, "w") as keras_file:
            keras_file.write(f"{keras}\n")

    def create_layers(self, layers: dict):
        filepath = os.path.join(self.modeling, self._modeling_layers)
        with open(filepath, "w") as layers_file:
            json.dump(layers, layers_file)

    def remove_plan(self):
        filepath = os.path.join(self.modeling, self._modeling_plan)
        if os.path.isfile(filepath):
            os.remove(filepath)

    def remove_preview(self):
        filepath = os.path.join(self.modeling, self._modeling_preview)
        if os.path.isfile(filepath):
            os.remove(filepath)

    def remove_keras(self):
        filepath = os.path.join(self.modeling, self._modeling_keras)
        if os.path.isfile(filepath):
            os.remove(filepath)

    def remove_layers(self):
        filepath = os.path.join(self.modeling, self._modeling_layers)
        if os.path.isfile(filepath):
            os.remove(filepath)

    def remove_training(self):
        filepath = os.path.join(self.training, self._training_output)
        if os.path.isfile(filepath):
            os.remove(filepath)
        for filename in os.listdir(self.training):
            filepath = os.path.join(self.training, filename)
            if filename.endswith(".h5") and os.path.isfile(filepath):
                os.remove(filepath)

    def clear_modeling(self):
        self.remove_plan()
        self.remove_preview()
        self.remove_keras()
        self.remove_layers()


class TerraExchangeProject(pydantic.BaseModel):
    tensorflow: str = ""
    error: str = ""
    name: str = "NoName"
    hardware: Hardware = Hardware.CPU
    datasets: List[Dataset] = []
    datasets_sources: List[str] = []
    tags: dict = {}
    dataset: str = ""
    model_name: str = ""
    model_plan: Optional[list] = []
    layers: Dict[int, Layer] = {}
    layers_start: Dict[int, Layer] = {}
    layers_schema: List[List[Any]] = []
    layers_types: Dict[str, LayerConfigParam] = {}
    optimizers: Dict[str, OptimizerParams] = {}
    callbacks: dict = {}
    compile: dict = {}
    training: TrainConfig = TrainConfig()
    in_training: bool = False
    path: dict = {
        "datasets": reverse_lazy("apps_project:datasets"),
        "modeling": reverse_lazy("apps_project:modeling"),
        "training": reverse_lazy("apps_project:training"),
    }
    dir: ProjectPath = ProjectPath()
    gd: GoogleDrivePath = GoogleDrivePath()

    def __init__(self, **kwargs):
        datasets = kwargs.get("datasets", [])
        kwargs["tensorflow"] = tensorflow.__version__
        super().__init__(**kwargs)

        if not os.path.isfile(self.dir.config):
            os.makedirs(settings.TERRA_AI_PROJECT_PATH, exist_ok=True)
            with open(self.dir.config, "w") as config_ref:
                config_ref.write("{}")
                config_ref.close()

        with open(self.dir.config, "r") as file:
            try:
                kwargs.update(**json.load(file))
            except Exception:
                pass

        kwargs["datasets"] = datasets
        super().__init__(**kwargs)

    def dict(self, *args, **kwargs):
        output = super().dict(*args, **kwargs)
        output["path"] = {
            "datasets": str(self.path.get("datasets", "")),
            "modeling": str(self.path.get("modeling", "")),
            "training": str(self.path.get("training", "")),
        }
        return output

    @property
    def model_validated(self) -> bool:
        return self.dir.validated

    def autosave(self):
        with open(self.dir.config, "w") as file:
            json.dump(self.dict(), file)

    def clear(self):
        shutil.rmtree(settings.TERRA_AI_PROJECT_PATH)

    def save_training_files(self, name: str):
        dir_path = os.path.join(self.gd.projects, name)
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)

        keras_path = os.path.join(self.dir.modeling, self.dir._modeling_keras)
        if os.path.isfile(keras_path):
            shutil.copy2(keras_path, os.path.join(dir_path, self.dir._modeling_keras))

        # if os.path.isfile(self.dir.config):
        #     shutil.copy2(self.dir.config, os.path.join(dir_path, "project.conf"))

        for item in os.listdir(self.dir.training):
            h5_path = os.path.join(self.dir.training, item)
            if os.path.isfile(h5_path) and item.endswith(".h5"):
                shutil.copy2(h5_path, os.path.join(dir_path, item))


@dataclass
class TerraExchangeResponse:
    success: bool
    error: str
    data: dict
    stop_flag: bool
    tb: list

    def __init__(self, **kwargs):
        self.success = kwargs.get("success", True)
        self.error = kwargs.get("error", "")
        self.data = kwargs.get("data", {})
        self.stop_flag = kwargs.get("stop_flag", True)
        self.tb = kwargs.get("tb", [])
