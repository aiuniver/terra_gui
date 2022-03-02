import os
import re
import json
import collections

from pathlib import Path, PureWindowsPath
from pandas import read_csv
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from pydantic import validator, DirectoryPath, PrivateAttr
from pydantic.types import PositiveInt
from pydantic.color import Color

from terra_ai.data.mixins import AliasMixinData, UniqueListMixin, BaseMixinData
from terra_ai.data.extra import FileSizeData, FileLengthData
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

from terra_ai.settings import (
    TERRA_PATH,
    DATASET_EXT,
    DATASET_CONFIG,
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
    col_type: Optional[Any]
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
    length: Optional[FileLengthData]


class DatasetVersionExtData(DatasetVersionData):
    inputs: Dict[PositiveInt, DatasetInputsData] = {}
    outputs: Dict[PositiveInt, DatasetOutputsData] = {}
    service: Dict[PositiveInt, DatasetOutputsData] = {}
    columns: Dict[PositiveInt, Dict[str, Any]] = {}


class DatasetData(AliasMixinData):
    name: str
    architecture: ArchitectureChoice
    tags: List[str] = []
    version: DatasetVersionExtData
    group: DatasetGroupChoice

    _path: Path = PrivateAttr()

    def __init__(self, **data):
        self._path = Path(data.get("path", ""))
        super().__init__(**data)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def inputs(self) -> Dict[PositiveInt, DatasetInputsData]:
        return self.version.inputs

    @property
    def outputs(self) -> Dict[PositiveInt, DatasetOutputsData]:
        return self.version.outputs

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


class TagsGroupData(BaseMixinData):
    name: ArchitectureChoice
    items: List[str]


class TagsGroupList(UniqueListMixin):
    class Meta:
        source = TagsGroupData
        identifier = "name"


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
    def tags(self) -> TagsGroupList:
        tags = []
        groups = {}
        for group in self:
            for dataset in group.datasets:
                if groups.get(dataset.architecture, None) is None:
                    groups[dataset.architecture] = []
                groups[dataset.architecture] += dataset.tags
        for architecture, group in groups.items():
            group = list(set(group))
            group.sort()
            groups[architecture] = group
        groups = collections.OrderedDict(sorted(groups.items()))
        return TagsGroupList(
            list(
                map(
                    lambda item: {"name": item[0], "items": item[1]},
                    groups.items(),
                )
            )
        )


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


class DatasetInfo(BaseMixinData):
    alias: str
    group: DatasetGroupChoice
