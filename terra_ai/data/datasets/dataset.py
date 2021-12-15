import os
import json

from pathlib import Path, PureWindowsPath
from pandas import read_csv
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from pydantic import validator, DirectoryPath, PrivateAttr
from pydantic.types import PositiveInt
from pydantic.color import Color

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
from terra_ai.data.presets.datasets import OutputLayersDefaults, DatasetsGroups

from terra_ai.exceptions.datasets import (
    DatasetUndefinedGroupException,
    DatasetNotFoundInGroupException,
    DatasetUndefinedConfigException,
)
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, DATASETS_LOADED_DIR


class DatasetLoadData(BaseMixinData):
    path: DirectoryPath
    group: DatasetGroupChoice
    alias: str

    @validator("alias")
    def _validate_alias(cls, value: str, values) -> str:
        group = values.get("group")

        if group in (DatasetGroupChoice.keras, DatasetGroupChoice.terra):
            groups = list(
                filter(lambda item: item.get("alias") == group.name, DatasetsGroups)
            )
            if not len(groups):
                raise DatasetUndefinedGroupException(group.value)
            dataset_match = list(
                filter(
                    lambda item: item.get("alias") == value,
                    groups[0].get("datasets", []),
                )
            )
            if not len(dataset_match):
                raise DatasetNotFoundInGroupException(value, group.value)

        elif group in (DatasetGroupChoice.custom,):
            dataset_path = Path(values.get("path"), f"{value}.{DATASET_EXT}")
            if not dataset_path.is_dir():
                raise DatasetNotFoundInGroupException(value, group.value)

            config_path = Path(dataset_path, DATASET_CONFIG)
            if not config_path.is_file():
                raise DatasetUndefinedConfigException(value, group.value)

        return value


class CustomDatasetConfigData(BaseMixinData):
    """
    Загрузка конфигурации пользовательского датасета
    """

    path: DirectoryPath
    config: Optional[dict] = {}

    # @validator("path")
    # def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
    #     if not str(value).endswith(f".{settings.DATASET_EXT}"):
    #         raise TrdsDirExtException(value.name)
    #     return value

    @validator("config", always=True)
    def _validate_config(cls, value: dict, values) -> dict:
        config_path = Path(values.get("path"), DATASET_CONFIG)
        if not config_path.is_file():
            raise TrdsConfigFileNotFoundException(
                values.get("path").name, config_path.name
            )
        with open(config_path, "r") as config_ref:
            value = json.load(config_ref)
        value.update({"group": DatasetGroupChoice.custom})
        return value


class DatasetLayerData(BaseMixinData):
    name: str
    datatype: str
    dtype: str
    shape: Tuple[PositiveInt, ...]
    num_classes: Optional[PositiveInt]
    classes_names: Optional[List[str]]
    classes_colors: Optional[List[Color]]
    encoding: LayerEncodingChoice = LayerEncodingChoice.none


class DatasetPathsData(BaseMixinData):
    basepath: DirectoryPath

    arrays: Optional[DirectoryPath]
    sources: Optional[DirectoryPath]
    instructions: Optional[DirectoryPath]
    preprocessing: Optional[DirectoryPath]

    @validator(
        "arrays",
        "sources",
        "instructions",
        "preprocessing",
        always=True,
    )
    def _validate_internal_path(cls, value, values, field) -> Path:
        path = Path(values.get("basepath"), field.name)
        os.makedirs(path, exist_ok=True)
        return path


class DatasetInputsData(DatasetLayerData):
    task: LayerInputTypeChoice


class DatasetOutputsData(DatasetLayerData):
    task: LayerOutputTypeChoice


class DatasetData(AliasMixinData):
    """
    Информация о датасете
    """

    name: str
    date: Optional[datetime]
    size: Optional[FileSizeData]
    group: Optional[DatasetGroupChoice]
    use_generator: bool = False
    architecture: ArchitectureChoice = ArchitectureChoice.Basic
    tags: Optional[TagsList] = TagsList()
    inputs: Optional[Dict[PositiveInt, DatasetInputsData]] = {}
    outputs: Optional[Dict[PositiveInt, DatasetOutputsData]] = {}
    service: Optional[Dict[PositiveInt, DatasetOutputsData]] = {}
    columns: Optional[Dict[PositiveInt, Dict[str, Any]]] = {}

    _path: Path = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        _path = data.get("path")
        if _path:
            self.set_path(_path)

    @property
    def path(self):
        return self._path

    @property
    def training_available(self) -> bool:
        return self.architecture != ArchitectureChoice.Tracker or self.outputs != {}

    @property
    def sources(self) -> List[str]:
        out = []
        sources = read_csv(Path(self.path, "instructions", "tables", "val.csv"))
        for column in sources.columns:
            _title = column.split("_")[-1].title()
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

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        data.update({"training_available": self.training_available})
        return data

    def set_path(self, value):
        self._path = Path(value)


class DatasetsList(UniqueListMixin):
    """
    Список датасетов, основанных на `DatasetData`
    ```
    class Meta:
        source = DatasetData
        identifier = "alias"
    ```
    """

    class Meta:
        source = DatasetData
        identifier = "alias"


class DatasetsGroupData(AliasMixinData):
    """
    Группа датасетов
    """

    name: str
    datasets: DatasetsList = DatasetsList()

    @property
    def tags(self) -> TagsList:
        __tags = TagsList()
        for item in self.datasets:
            __tags += item.tags
        return __tags

    def dict(self, **kwargs):
        data = super().dict()
        data.update({"tags": self.tags.dict()})
        return data


class DatasetsGroupsList(UniqueListMixin):
    """
    Список групп датасетов, основанных на `DatasetsGroupData`
    ```
    class Meta:
        source = DatasetsGroupData
        identifier = "alias"
    ```
    """

    class Meta:
        source = DatasetsGroupData
        identifier = "alias"


class DatasetInfo(BaseMixinData):
    alias: str
    group: DatasetGroupChoice

    __dataset__: Optional[DatasetData] = PrivateAttr()

    def __init__(self, **data):
        self.__dataset__ = None
        super().__init__(**data)

    @property
    def dataset(self) -> Optional[DatasetData]:
        if not self.__dataset__:
            config_path = Path(
                DATASETS_LOADED_DIR,
                self.group.name,
                self.alias,
                DATASET_CONFIG,
            )
            if config_path.is_file():
                with open(config_path) as config_ref:
                    self.__dataset__ = DatasetData(
                        path=Path(
                            DATASETS_LOADED_DIR,
                            self.group.name,
                            self.alias,
                        ),
                        **json.load(config_ref),
                    )
        return self.__dataset__
