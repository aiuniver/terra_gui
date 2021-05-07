import json

import pydantic

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from django.urls import reverse_lazy


class Color(str, Enum):
    red = "\033[0;31m"
    reset = "\033[0m"


class OptimizerType(str, Enum):
    SGD = "SGD"
    RMSpro = "RMSprop"
    Adam = "Adam"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"


class OptimizerParams(pydantic.BaseModel):
    params: Dict[str, Optional[Any]] = {}



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
                if isinstance(item, dict):
                    item = json.dumps(item)
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
    input_shape: List[int] = []
    output_shape: List[int] = []
    data_name: str = ""
    data_available: List[str] = []
    params: LayerConfigParam = LayerConfigParam()

    @property
    def as_dict(self) -> dict:
        output = dict(self)
        output["type"] = self.type.value
        output["location_type"] = self.location_type.value
        output["params"] = dict(self.params)
        return output

    @pydantic.validator("up_link", allow_reuse=True)
    def correct_list_natural_number(cls, value):
        value = list(filter(lambda value: value > 0, value))
        value = list(set(value))
        return value

    @pydantic.validator("input_shape", "output_shape", allow_reuse=True)
    def correct_shape(cls, value):
        value = list(filter(lambda value: value > 0, value))
        return value


class Layer(pydantic.BaseModel):
    x: Optional[float]
    y: Optional[float]
    down_link: List[int] = []
    config: LayerConfig = LayerConfig()

    @property
    def as_dict(self) -> dict:
        output = dict(self)
        output["config"] = self.config.as_dict
        return output

    @pydantic.validator("down_link", allow_reuse=True)
    def validate_list_natural_number(cls, value):
        value = list(filter(lambda value: value > 0, value))
        value = list(set(value))
        return value


class LayerDict(pydantic.BaseModel):
    items: Dict[int, Layer] = {}

    @property
    def as_dict(self) -> dict:
        output = {"items": {}}
        for index, item in self.items.items():
            output["items"][index] = item.as_dict
        return output

    def reset_indexes(self):
        layers_rels = {}

        def _prepare(num: int = 0, update: list = None):
            update_next = []

            if update:
                for index in update:
                    num += 1
                    layers_rels[num] = int(index)
                for index, layer in self.items.items():
                    if list(set(update) & set(layer.config.up_link)):
                        update_next.append(int(index))
            else:
                for index, layer in self.items.items():
                    if not layer.config.up_link:
                        num += 1
                        layers_rels[num] = int(index)
                        update_next += layer.down_link

            update_next = list(set(update_next))
            if update_next:
                _prepare(num, update_next)

        _prepare()

        layers = {}
        for index, rel in layers_rels.items():
            layer = self.items.get(int(rel))
            layer.down_link = list(
                map(
                    lambda value: int(
                        list(layers_rels.keys())[
                            list(layers_rels.values()).index(int(value))
                        ]
                    ),
                    layer.down_link,
                )
            )
            layer.config.up_link = list(
                map(
                    lambda value: int(
                        list(layers_rels.keys())[
                            list(layers_rels.values()).index(int(value))
                        ]
                    ),
                    layer.config.up_link,
                )
            )
            layers[index] = layer

        self.items = layers


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


@dataclass
class TerraExchangeProject:
    error: str
    name: str
    hardware: str
    datasets: dict
    tags: dict
    dataset: str
    model_name: str
    layers: LayerDict
    start_layers: LayerDict
    schema: list
    layers_types: dict
    optimizers: list
    callbacks: dict
    compile: dict
    path: dict

    def __init__(self, **kwargs):
        self.error = kwargs.get("error", "")
        self.name = kwargs.get("name", "NoName")
        self.hardware = kwargs.get("hardware", "CPU")
        self.datasets = kwargs.get("datasets", {})
        self.tags = kwargs.get("tags", {})
        self.dataset = kwargs.get("dataset", "")
        self.model_name = kwargs.get("model_name", "")
        self.layers = kwargs.get("layers", LayerDict())
        self.start_layers = kwargs.get("start_layers", LayerDict())
        self.schema = kwargs.get("schema", [])
        self.layers_types = kwargs.get("layers_types", {})
        self.optimizers = kwargs.get("optimizers", [])
        self.callbacks = kwargs.get("callbacks", {})
        self.compile = kwargs.get("compile", {})
        self.path = {
            "datasets": reverse_lazy("apps_project:datasets"),
            "modeling": reverse_lazy("apps_project:modeling"),
            "training": reverse_lazy("apps_project:training"),
        }

    def __repr__(self):
        return f"""TerraExchangeProject:
    error        : {True if self.error else False}
    name         : {self.name}
    hardware     : {self.hardware}
    datasets     : {len(self.datasets.keys())}
    tags         : {len(self.tags.keys())}
    dataset      : {self.dataset}
    model_name   : {self.model_name}
    layers       : {len(self.layers.items.keys())}
    start_layers : {len(self.start_layers.items.keys())}
    schema       : {len(self.schema)}
    layers_types : {len(self.layers_types.keys())}
    optimizers   : {len(self.optimizers)}
    callbacks    : {len(self.callbacks.keys())}
    compile      : {len(self.compile.keys())}
    path         : datasets -> {self.path.get("modeling", f"{Color.red}undefined{Color.reset}")}
                   modeling -> {self.path.get("modeling", f"{Color.red}undefined{Color.reset}")}
                   training -> {self.path.get("training", f"{Color.red}undefined{Color.reset}")}"""

    @property
    def dataset_selected(self) -> bool:
        return self.dataset != ""

    @property
    def as_json_string(self) -> dict:
        output = dict(self.__dict__)
        path = {}
        for name, value in self.path.items():
            path.update({name: str(value)})
        output.update(
            {
                "path": path,
                "dataset_selected": self.dataset_selected,
                "layers": self.layers.as_dict.get("items"),
                "start_layers": self.start_layers.as_dict.get("items"),
            }
        )
        return output
