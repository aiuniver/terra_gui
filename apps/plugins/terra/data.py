import os
import json
import yaml
import cairosvg
import pydantic

from enum import Enum
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

from django.conf import settings
from django.urls import reverse_lazy


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
    save_best: bool = False
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
    num_classes: int = 2
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


class GoogleDrivePath(pydantic.BaseModel):
    datasets: str = f"{settings.TERRA_AI_DATA_PATH}/datasets"
    modeling: str = f"{settings.TERRA_AI_DATA_PATH}/modeling"
    training: str = f"{settings.TERRA_AI_DATA_PATH}/training"
    projects: str = f"{settings.TERRA_AI_DATA_PATH}/projects"


class ProjectPath(pydantic.BaseModel):
    datasets: str = f"{settings.TERRA_AI_PROJECT_PATH}/datasets"
    modeling: str = f"{settings.TERRA_AI_PROJECT_PATH}/modeling"
    training: str = f"{settings.TERRA_AI_PROJECT_PATH}/training"
    config: str = f"{settings.TERRA_AI_PROJECT_PATH}/project.conf"

    _modeling_plan = "plan.yaml"
    _modeling_preview = "preview.png"
    _modeling_keras = "keras.py"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.datasets, exist_ok=True)
        os.makedirs(self.modeling, exist_ok=True)
        os.makedirs(self.training, exist_ok=True)

    @property
    def model_validated(self) -> bool:
        return (
            os.path.isfile(f"{self.modeling}/{self._modeling_plan}")
            and os.path.isfile(f"{self.modeling}/{self._modeling_preview}")
            and os.path.isfile(f"{self.modeling}/{self._modeling_keras}")
        )

    @property
    def is_keras(self) -> bool:
        return os.path.isfile(f"{self.modeling}/{self._modeling_keras}")

    @property
    def keras_code(self) -> (bool, str):
        success = False
        output = ""
        try:
            with open(f"{self.modeling}/{self._modeling_keras}", "r") as keras_file:
                success = True
                output = keras_file.read()
        except Exception as error:
            output = str(error)
        return success, output

    def save_modeling(self, svg: str, yaml_info: dict, keras: str):
        with open(f"{self.modeling}/{self._modeling_plan}", "w") as yaml_file:
            yaml.dump(yaml_info, yaml_file)
        with open(f"{self.modeling}/{self._modeling_keras}", "w") as keras_file:
            keras_file.write(f"{keras}\n")
        cairosvg.svg2png(svg, write_to=f"{self.modeling}/{self._modeling_preview}")

    def clear_modeling(self):
        if os.path.isfile(f"{self.modeling}/{self._modeling_plan}"):
            os.remove(f"{self.modeling}/{self._modeling_plan}")
        if os.path.isfile(f"{self.modeling}/{self._modeling_keras}"):
            os.remove(f"{self.modeling}/{self._modeling_keras}")
        if os.path.isfile(f"{self.modeling}/{self._modeling_preview}"):
            os.remove(f"{self.modeling}/{self._modeling_preview}")


class TerraExchangeProject(pydantic.BaseModel):
    error: str = ""
    name: str = "NoName"
    hardware: Hardware = Hardware.CPU
    datasets: List[Dataset] = []
    tags: dict = {}
    dataset: str = ""
    model_name: str = ""
    model_plan: Optional[list] = []
    layers: Dict[int, Layer] = {}
    layers_start: Dict[int, Layer] = {}
    layers_schema: List[List[int]] = []
    layers_types: Dict[str, LayerConfigParam] = {}
    optimizers: Dict[str, OptimizerParams] = {}
    callbacks: dict = {}
    compile: dict = {}
    training: TrainConfig = TrainConfig()
    path: dict = {
        "datasets": reverse_lazy("apps_project:datasets"),
        "modeling": reverse_lazy("apps_project:modeling"),
        "training": reverse_lazy("apps_project:training"),
    }
    dir: ProjectPath = ProjectPath()
    gd: GoogleDrivePath = GoogleDrivePath()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not os.path.isfile(self.dir.config):
            open(self.dir.config, "a").close()

        with open(self.dir.config, "r") as file:
            try:
                kwargs.update(**json.load(file))
            except Exception:
                pass

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
        return self.dir.model_validated

    def autosave(self):
        with open(self.dir.config, "w") as file:
            json.dump(self.dict(), file)


@dataclass
class TerraExchangeResponse:
    success: bool
    error: str
    data: dict
    stop_flag: bool

    def __init__(self, **kwargs):
        self.success = kwargs.get("success", True)
        self.error = kwargs.get("error", "")
        self.data = kwargs.get("data", {})
        self.stop_flag = kwargs.get("stop_flag", True)
