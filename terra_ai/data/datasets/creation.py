from math import fsum
from pathlib import Path
from typing import Union, Optional, Any, Dict, List, Tuple
from pydantic import validator
from pydantic.types import DirectoryPath, PositiveInt
from pydantic.networks import HttpUrl
from pydantic.errors import EnumMemberError

from terra_ai import settings as terra_ai_settings
from terra_ai.data.datasets.extra import DatasetTaskTypeChoice
from terra_ai.data.mixins import (
    BaseMixinData,
    UniqueListMixin,
    AliasMixinData,
    IDMixinData,
)
from terra_ai.data.types import (
    confilepath,
    confilename,
    FilePathType,
    ConstrainedFloatValueGe0Le1,
    ConstrainedLayerNameValue,
)
from terra_ai.data.exceptions import (
    ValueTypeException,
    PartTotalException,
    ListEmptyException,
    ObjectDetectionQuantityLayersException,
)
from terra_ai.data.datasets.extra import (
    SourceModeChoice,
    LayerTypeChoice,
    LayerGroupChoice,
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerDatatypeChoice,
    ColumnProcessingTypeChoice,
)
from terra_ai.data.datasets.creations import column_processing
from terra_ai.data.datasets.tags import TagsList
from terra_ai.data.datasets import creations
from terra_ai.data.training.extra import ArchitectureChoice


class FilePathSourceData(BaseMixinData):
    value: confilepath(ext="zip")
    label: Optional[str]

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = values.get("value")
        if not file_path:
            return value
        return file_path.name.split(".zip")[0]


class FilePathSourcesList(UniqueListMixin):
    class Meta:
        source = FilePathSourceData
        identifier = "label"


class SourceData(BaseMixinData):
    """
    Информация для загрузки исходников датасета
    """

    mode: SourceModeChoice
    "Режим загрузки исходных данных"
    value: Union[confilepath(ext="zip"), HttpUrl, confilename(ext="zip")]
    "Значение для режим загрузки исходных данных. Тип будет зависеть от выбранного режима `mode`"

    @validator("value", allow_reuse=True)
    def _validate_mode_value(
        cls, value: Union[FilePathType, HttpUrl, str], values
    ) -> Union[FilePathType, HttpUrl, str]:
        mode = values.get("mode")
        if mode == SourceModeChoice.GoogleDrive:
            if not isinstance(value, Path):
                raise ValueTypeException(value, FilePathType)
        if mode == SourceModeChoice.URL:
            if not isinstance(value, HttpUrl):
                raise ValueTypeException(value, HttpUrl)
        return value


class CreationInfoPartData(BaseMixinData):
    """
    Доли использования данных для обучающей, тестовой и валидационной выборок"
    """

    train: ConstrainedFloatValueGe0Le1 = 0.7
    "Обучающая выборка"
    validation: ConstrainedFloatValueGe0Le1 = 0.3
    "Валидационная выборка"

    @property
    def total(self) -> float:
        """
        Сумма всех значений
        """
        return fsum([self.train, self.validation])


class CreationInfoData(BaseMixinData):
    """
    Информация о данных датасета
    """

    part: CreationInfoPartData = CreationInfoPartData()
    "Доли выборок"
    shuffle: Optional[bool] = True
    "Случайным образом перемешивать элементы"

    @validator("part", allow_reuse=True)
    def _validate_part(cls, value: CreationInfoPartData) -> CreationInfoPartData:
        if value.total != float(1):
            raise PartTotalException(value)
        return value


class CreationParametersData(BaseMixinData):

    sources_paths: Optional[list]
    names: Dict[str, list]


class CreationInputData(IDMixinData):
    """
    Информация о `input`-слое
    """

    name: ConstrainedLayerNameValue
    "Название"
    parameters: Dict[str, Dict[str, list]]
    "Параметры"
    # type: #LayerInputTypeChoice
    # "Тип данных"
    # parameters: Any
    # "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    # @validator("type", pre=True)
    # def _validate_type(cls, value: LayerInputTypeChoice) -> LayerInputTypeChoice:
    #     if value not in list(LayerInputTypeChoice):
    #         raise EnumMemberError(enum_values=list(LayerInputTypeChoice))
    #     name = (
    #         value
    #         if isinstance(value, LayerInputTypeChoice)
    #         else LayerInputTypeChoice(value)
    #     ).name
    #     type_ = getattr(
    #         creations.layers.input, getattr(creations.layers.input.Layer, name)
    #     )
    #     cls.__fields__["parameters"].type_ = type_
    #     return value

    # @validator("parameters", always=True)
    # def _validate_parameters(cls, value: Any, values, field) -> Any:
    #     return field.type_(**value or {})


class CreationOutputData(IDMixinData):
    """
    Информация о `output`-слое
    """

    name: ConstrainedLayerNameValue
    "Название"
    parameters: Dict[str, Dict[str, list]]
    "Параметры"

    # type: #LayerOutputTypeChoice
    # "Тип данных"
    # parameters: Any
    # "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    # @validator("type", pre=True)
    # def _validate_type(cls, value: LayerOutputTypeChoice) -> LayerOutputTypeChoice:
    #     if value not in list(LayerOutputTypeChoice):
    #         raise EnumMemberError(enum_values=list(LayerOutputTypeChoice))
    #     name = (
    #         value
    #         if isinstance(value, LayerOutputTypeChoice)
    #         else LayerOutputTypeChoice(value)
    #     ).name
    #     type_ = getattr(
    #         creations.layers.output, getattr(creations.layers.output.Layer, name)
    #     )
    #     cls.__fields__["parameters"].type_ = type_
    #     return value

    # @validator("parameters", always=True)
    # def _validate_parameters(cls, value: Any, values, field) -> Any:
    #     return field.type_(**value or {})


class CreationInputsList(UniqueListMixin):
    """
    Список `input`-слоев, основанных на `CreationInputData`
    ```
    class Meta:
        source = CreationInputData
        identifier = "alias"
    ```
    """

    class Meta:
        source = CreationInputData
        identifier = "id"


class CreationOutputsList(UniqueListMixin):
    """
    Список `output`-слоев, основанных на `CreationOutputData`
    ```
    class Meta:
        source = CreationOutputData
        identifier = "alias"
    ```
    """

    class Meta:
        source = CreationOutputData
        identifier = "id"


class ColumnsProcessingData(BaseMixinData):
    type: ColumnProcessingTypeChoice
    parameters: Any

    @validator("type", pre=True)
    def _validate_type(
        cls, value: ColumnProcessingTypeChoice
    ) -> ColumnProcessingTypeChoice:
        if not value:
            return value
        name = (
            value
            if isinstance(value, ColumnProcessingTypeChoice)
            else ColumnProcessingTypeChoice(value)
        ).name
        cls.__fields__["parameters"].type_ = getattr(
            column_processing, column_processing.ColumnProcessing[name].value
        )
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**(value or {}))


class CreationVersionData(AliasMixinData):
    """
    Полная информация о создании версии датасета
    Inputs:
        alias: str - alias версии
        name: str - название версии
        datasets_path: pathlib.Path - путь к директории датасетов проекта (./TerraAI/datasets)
        parent_alias: str - alias датасета, от которого создаётся версия
        info: CreationInfoData - соотношение обучающей выборки к валидационной, а также его перемешивание
        use_generator: bool - использовать генератор при обучении. По умолчанию: False
        inputs: CreationInputsList - входные слои
        outputs: CreationOutputsList - выходные слои
    """

    name: str
    "Название"
    datasets_path: DirectoryPath
    "Путь к директории датасетов проекта"
    parent_alias: str
    "Алиас родительского датасета"
    info: CreationInfoData = CreationInfoData()  # Train/Val split, shuffle
    "Информация о данных"
    processing: Dict[str, ColumnsProcessingData] = {}
    "Обработчики"
    inputs: CreationInputsList = CreationInputsList()
    "Входные слои"
    outputs: CreationOutputsList = CreationOutputsList()
    "Выходные слои"


class CreationData(AliasMixinData):
    """
    Полная информация о создании датасета
    """

    name: str
    source: SourceData
    architecture: ArchitectureChoice
    tags: List[str]
    version: Optional[CreationVersionData]  # Optional больше сделано для дебаггинга

    @property
    def path(self) -> Path:
        return Path(self.datasets_path, f"{self.alias}.{terra_ai_settings.DATASET_EXT}")

    # @validator("inputs", "outputs")
    # def _validate_required(cls, value: UniqueListMixin) -> UniqueListMixin:
    #     if not len(value):
    #         raise ListEmptyException(type(value))
    #     return value
    #
    # @validator("outputs")
    # def _validate_outputs(cls, value: UniqueListMixin) -> UniqueListMixin:
    #     if not value:
    #         return value
    #     is_object_detection = False
    #     for layer in value:
    #         if layer.type == LayerOutputTypeChoice.ObjectDetection:
    #             is_object_detection = True
    #             break
    #     if is_object_detection and len(value) > 1:
    #         raise ObjectDetectionQuantityLayersException(f"{len(value)} output layers")
    #     return value


class LayerBindData(BaseMixinData):
    up: List[PositiveInt] = []
    down: List[PositiveInt] = []


class CreationBlockData(IDMixinData):
    name: str
    type: LayerTypeChoice
    removable: bool = False
    bind: LayerBindData
    position: Tuple[int, int]
    datatype: Optional[LayerDatatypeChoice]
    parameters: Any


class CreationBlockList(UniqueListMixin):
    class Meta:
        source = CreationBlockData
        identifier = "id"


class CreationValidateBlocksData(BaseMixinData):
    type: LayerGroupChoice
    items: List[CreationBlockData]


class DatasetCreationArchitectureData(BaseMixinData):
    inputs: CreationBlockList
    outputs: CreationBlockList
