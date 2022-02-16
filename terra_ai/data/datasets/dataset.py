import os
import re
import json
import itertools

from copy import deepcopy
from pathlib import Path, PureWindowsPath
from pandas import read_csv
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from pydantic import validator, DirectoryPath, PrivateAttr
from pydantic.types import PositiveInt
from pydantic.color import Color
from dict_recursive_update import recursive_update

from terra_ai.data.mixins import AliasMixinData, UniqueListMixin, BaseMixinData
from terra_ai.data.extra import FileSizeData
from terra_ai.data.exceptions import TrdsConfigFileNotFoundException
from terra_ai.data.datasets.tags import TagsList
from terra_ai.data.datasets.extra import (
    DatasetGroupChoice,
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerEncodingChoice,
)
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.modeling.extra import LayerTypeChoice, LayerGroupChoice
from terra_ai.data.modeling.layers.extra import ActivationChoice
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.data.presets.models import EmptyModelDetailsData
from terra_ai.data.presets.datasets import OutputLayersDefaults, DatasetCommonGroup

from terra_ai.exceptions.datasets import (
    DatasetUndefinedGroupException,
    DatasetNotFoundInGroupException,
    DatasetUndefinedConfigException,
)
from terra_ai.settings import (
    TERRA_PATH,
    DATASET_EXT,
    DATASET_CONFIG,
    DATASETS_LOADED_DIR,
    DATASET_VERSION_EXT,
)


class DatasetLoadData(BaseMixinData):
    path: DirectoryPath
    group: DatasetGroupChoice
    alias: str
    version: str


class DatasetLayerData(BaseMixinData):
    name: str
    datatype: str
    dtype: str
    shape: Tuple[PositiveInt, ...]
    num_classes: Optional[PositiveInt]
    classes_names: Optional[List[str]]
    classes_colors: Optional[List[Color]]
    encoding: LayerEncodingChoice = LayerEncodingChoice.none


class DatasetInputsData(DatasetLayerData):
    task: LayerInputTypeChoice


class DatasetOutputsData(DatasetLayerData):
    task: LayerOutputTypeChoice


class DatasetVersionData(AliasMixinData):
    name: str
    date: Optional[datetime]
    size: Optional[FileSizeData]


class DatasetVersionExtData(AliasMixinData):
    name: str


class DatasetData(DatasetVersionData):
    architecture: ArchitectureChoice
    tags: List[str] = []
    version: DatasetVersionExtData
    group: DatasetGroupChoice
    use_generator: bool = False
    inputs: Dict[PositiveInt, DatasetInputsData] = {}
    outputs: Dict[PositiveInt, DatasetOutputsData] = {}
    service: Dict[PositiveInt, DatasetOutputsData] = {}
    columns: Dict[PositiveInt, Dict[str, Any]] = {}

    _path: Path = PrivateAttr()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def model(self) -> ModelDetailsData:
        data = {**EmptyModelDetailsData}
        layers = []
        for _id, layer in self.inputs.items():
            _data = {
                "id": _id,
                "name": layer.name,
                "type": LayerTypeChoice.Input,
                "group": LayerGroupChoice.input,
                "shape": {"input": [layer.shape]},
                "task": layer.task,
            }
            if layer.num_classes:
                _data.update(
                    {
                        "num_classes": layer.num_classes,
                    }
                )
            layers.append(_data)
        for _id, layer in self.outputs.items():
            output_layer_defaults = OutputLayersDefaults.get(layer.task, {}).get(
                layer.datatype, {}
            )
            activation = output_layer_defaults.get("activation", ActivationChoice.relu)
            units = layer.num_classes
            params = {
                "activation": activation,
            }
            if units:
                params.update(
                    {
                        "units": units,
                        "filters": units,
                    }
                )
            _data = {
                "id": _id,
                "name": layer.name,
                "type": output_layer_defaults.get("type", LayerTypeChoice.Dense),
                "group": LayerGroupChoice.output,
                "shape": {"output": [layer.shape]},
                "task": layer.task,
                "parameters": {
                    "main": params,
                    "extra": params,
                },
            }
            if layer.num_classes:
                _data.update(
                    {
                        "num_classes": layer.num_classes,
                    }
                )
            layers.append(_data)
        data.update({"layers": layers})
        return ModelDetailsData(**data)

    @property
    def training_available(self) -> bool:
        return self.architecture not in (
            ArchitectureChoice.VideoTracker,
            ArchitectureChoice.Speech2Text,
            ArchitectureChoice.Text2Speech,
        )

    @property
    def sources(self) -> List[str]:
        out = []
        sources = read_csv(Path(self.path, "instructions", "tables", "val.csv"))
        for column in sources.columns:
            match = re.findall(r"^([\d]+_)(.+)$", column)
            if not match:
                continue
            _title = match[0][1].title()
            if _title in ["Image", "Text", "Audio", "Video"]:
                out = sources[column].to_list()
                if _title != "Text":
                    out = list(
                        map(
                            lambda item: str(
                                Path(self.path, PureWindowsPath(item).as_posix())
                            ),
                            out,
                        )
                    )
        return out


class DatasetVersionList(UniqueListMixin):
    class Meta:
        source = DatasetVersionData
        identifier = "alias"

    def __init__(self, group: str, alias: str):
        if group == "custom":
            versions = []
            dataset_path = DatasetCommonPathsData(
                basepath=Path(TERRA_PATH.datasets, f"{alias}.{DATASET_EXT}")
            )
            for item in dataset_path.versions.iterdir():
                if item.suffix[1:] != DATASET_VERSION_EXT or not item.is_dir():
                    continue
                try:
                    with open(Path(item, "version.json")) as version_ref:
                        versions.append(json.load(version_ref))
                except Exception:
                    pass
            args = (versions,)
        else:
            datasets = DatasetCommonGroupList()
            try:
                args = (datasets.get(group).datasets.get(alias).versions,)
            except Exception:
                args = ()
        super().__init__(*args)


class DatasetCommonData(AliasMixinData):
    name: str
    architecture: ArchitectureChoice = ArchitectureChoice.Basic
    tags: List[str] = []

    __group__: DatasetGroupChoice = PrivateAttr()
    __versions__: List[dict] = PrivateAttr()

    def __init__(self, **data):
        self.__group__ = data.pop("group", "")
        self.__versions__ = data.pop("versions", [])
        super().__init__(**data)

    @property
    def id(self) -> str:
        return f"{self.group}_{self.alias}"

    @property
    def group(self) -> DatasetGroupChoice:
        return self.__group__

    @property
    def versions(self) -> List[dict]:
        return self.__versions__

    def dict(self, **kwargs) -> dict:
        data = super().dict(**kwargs)
        data.update(
            {
                "id": self.id,
                "group": self.group,
            }
        )
        return data


class DatasetCommonList(UniqueListMixin):
    class Meta:
        source = DatasetCommonData
        identifier = "alias"


class DatasetCommonGroupData(AliasMixinData):
    name: str
    datasets: DatasetCommonList = DatasetCommonList()


class DatasetCommonGroupList(UniqueListMixin):
    class Meta:
        source = DatasetCommonGroupData
        identifier = "alias"

    def __init__(self):
        args = (DatasetCommonGroup,)
        custom = []
        for item in TERRA_PATH.datasets.iterdir():
            if item.suffix[1:] != DATASET_EXT or not item.is_dir():
                continue
            try:
                with open(Path(item, DATASET_CONFIG)) as config_ref:
                    custom.append(json.load(config_ref))
            except Exception:
                pass
        for item in args[0]:
            if item.get("alias") == "custom":
                item["datasets"] = custom
            for dataset in item.get("datasets"):
                dataset.update({"group": item.get("alias")})
        super().__init__(*args)

    @property
    def tags(self) -> List[str]:
        tags = []
        for group in self:
            tags += list(
                itertools.chain.from_iterable(
                    list(map(lambda item: item.tags, group.datasets))
                )
            )
        tags = list(set(tags))
        tags.sort()
        return tags


# class CustomDatasetConfigData(BaseMixinData):
#     """
#     Загрузка конфигурации пользовательского датасета
#     """
#
#     path: DirectoryPath
#     config: Optional[dict] = {}
#
#     # @validator("path")
#     # def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
#     #     if not str(value).endswith(f".{settings.DATASET_EXT}"):
#     #         raise TrdsDirExtException(value.name)
#     #     return value
#
#     @validator("config", always=True)
#     def _validate_config(cls, value: dict, values) -> dict:
#         config_path = Path(values.get("path"), DATASET_CONFIG)
#         if not config_path.is_file():
#             raise TrdsConfigFileNotFoundException(
#                 values.get("path").name, config_path.name
#             )
#         with open(config_path, "r") as config_ref:
#             value = json.load(config_ref)
#         value.update({"group": DatasetGroupChoice.custom})
#         return value


class DatasetCommonPathsData(BaseMixinData):
    basepath: DirectoryPath
    versions: Optional[DirectoryPath]
    sources: Optional[DirectoryPath]

    @validator("versions", "sources", always=True)
    def _validate_internal_path(cls, value, values, field) -> Path:
        path = Path(values.get("basepath"), field.name)
        os.makedirs(path, exist_ok=True)
        return path


class DatasetVersionPathsData(BaseMixinData):
    basepath: DirectoryPath
    arrays: Optional[DirectoryPath]
    sources: Optional[DirectoryPath]
    instructions: Optional[DirectoryPath]
    preprocessing: Optional[DirectoryPath]

    @validator("arrays", "sources", "instructions", "preprocessing", always=True)
    def _validate_internal_path(cls, value, values, field) -> Path:
        path = Path(values.get("basepath"), field.name)
        os.makedirs(path, exist_ok=True)
        return path


class VersionData(AliasMixinData):
    pass


# class VersionData(AliasMixinData):
#     """
#     Информация о версии
#     """
#
#     name: str
#     date: Optional[datetime]
#     size: Optional[FileSizeData]
#     inputs: Dict[PositiveInt, DatasetInputsData] = {}
#     outputs: Dict[PositiveInt, DatasetOutputsData] = {}
#     service: Dict[PositiveInt, DatasetOutputsData] = {}
#     columns: Dict[PositiveInt, Dict[str, Any]] = {}
#
#     @property
#     def model(self) -> ModelDetailsData:
#         data = {**EmptyModelDetailsData}
#         layers = []
#         for _id, layer in self.inputs.items():
#             _data = {
#                 "id": _id,
#                 "name": layer.name,
#                 "type": LayerTypeChoice.Input,
#                 "group": LayerGroupChoice.input,
#                 "shape": {"input": [layer.shape]},
#                 "task": layer.task,
#             }
#             if layer.num_classes:
#                 _data.update(
#                     {
#                         "num_classes": layer.num_classes,
#                     }
#                 )
#             layers.append(_data)
#         for _id, layer in self.outputs.items():
#             output_layer_defaults = OutputLayersDefaults.get(layer.task, {}).get(
#                 layer.datatype, {}
#             )
#             activation = output_layer_defaults.get("activation", ActivationChoice.relu)
#             units = layer.num_classes
#             params = {
#                 "activation": activation,
#             }
#             if units:
#                 params.update(
#                     {
#                         "units": units,
#                         "filters": units,
#                     }
#                 )
#             _data = {
#                 "id": _id,
#                 "name": layer.name,
#                 "type": output_layer_defaults.get("type", LayerTypeChoice.Dense),
#                 "group": LayerGroupChoice.output,
#                 "shape": {"output": [layer.shape]},
#                 "task": layer.task,
#                 "parameters": {
#                     "main": params,
#                     "extra": params,
#                 },
#             }
#             if layer.num_classes:
#                 _data.update(
#                     {
#                         "num_classes": layer.num_classes,
#                     }
#                 )
#             layers.append(_data)
#         data.update({"layers": layers})
#         return ModelDetailsData(**data)
#
#
# class DatasetData(AliasMixinData):
#     """
#     Информация о датасете
#     """
#
#     name: str
#     architecture: ArchitectureChoice = ArchitectureChoice.Basic
#     group: Optional[DatasetGroupChoice]
#     tags: Optional[TagsList] = TagsList()
#     version: Optional[VersionData] = None
#
#     _path: Path = PrivateAttr()
#
#     def __init__(self, **data):
#         super().__init__(**data)
#         _path = data.get("path")
#         if _path:
#             self.set_path(_path)
#
#     @property
#     def model(self) -> Optional[ModelDetailsData]:
#         return self.version.model if self.version else None
#
#     @property
#     def inputs(self) -> Dict[PositiveInt, DatasetInputsData]:
#         return self.version.inputs if self.version else {}
#
#     @property
#     def outputs(self) -> Dict[PositiveInt, DatasetOutputsData]:
#         return self.version.outputs if self.version else {}
#
#     @property
#     def path(self):
#         return self._path
#
#     @property
#     def training_available(self) -> bool:
#         return self.architecture not in (
#             ArchitectureChoice.VideoTracker,
#             ArchitectureChoice.Speech2Text,
#             ArchitectureChoice.Text2Speech,
#         )
#
#     @property
#     def sources(self) -> List[str]:
#         out = []
#         sources = read_csv(Path(self.path, "instructions", "tables", "val.csv"))
#         for column in sources.columns:
#             match = re.findall(r"^([\d]+_)(.+)$", column)
#             if not match:
#                 continue
#             _title = match[0][1].title()
#             if _title in ["Image", "Text", "Audio", "Video"]:
#                 out = sources[column].to_list()
#                 if _title != "Text":
#                     out = list(
#                         map(
#                             lambda item: str(
#                                 Path(self.path, PureWindowsPath(item).as_posix())
#                             ),
#                             out,
#                         )
#                     )
#         return out
#
#     def dict(self, **kwargs):
#         data = super().dict(**kwargs)
#         data.update({"training_available": self.training_available})
#         return data
#
#     def set_path(self, value):
#         self._path = Path(value)


class DatasetInfo(BaseMixinData):
    alias: str
    group: DatasetGroupChoice


# class DatasetInfo(BaseMixinData):
#     alias: str
#     group: DatasetGroupChoice
#
#     __dataset__: Optional[DatasetData] = PrivateAttr()
#
#     def __init__(self, **data):
#         self.__dataset__ = None
#         super().__init__(**data)
#
#     @property
#     def dataset(self) -> Optional[DatasetData]:
#         if not self.__dataset__:
#             config_path = Path(
#                 DATASETS_LOADED_DIR,
#                 self.group.name,
#                 self.alias,
#                 DATASET_CONFIG,
#             )
#             if config_path.is_file():
#                 with open(config_path) as config_ref:
#                     self.__dataset__ = DatasetData(
#                         path=Path(
#                             DATASETS_LOADED_DIR,
#                             self.group.name,
#                             self.alias,
#                         ),
#                         **json.load(config_ref),
#                     )
#         return self.__dataset__
