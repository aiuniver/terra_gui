import os
import re

from math import fsum
from pathlib import Path
from typing import Union, Optional, Any, Dict, List, Tuple
from transliterate import slugify

from pydantic import validator, DirectoryPath, PositiveInt, PrivateAttr
from pydantic.networks import HttpUrl
from pydantic.errors import EnumMemberError

from terra_ai import settings as terra_ai_settings
from terra_ai.data.mixins import (
    BaseMixinData,
    UniqueListMixin,
    AliasMixinData,
    IDMixinData,
)
from terra_ai.data.types import (
    AliasType,
    ConstrainedFloatValueGe0Le1,
    ConstrainedLayerNameValue,
)
from terra_ai.data.exceptions import ValueTypeException, PartTotalException
from terra_ai.data.datasets.extra import (
    SourceModeChoice,
    LayerTypeChoice,
    LayerGroupChoice,
    ColumnProcessingTypeChoice,
)

from terra_ai.data.datasets import creations
from terra_ai.data.training.extra import ArchitectureChoice


class SourceData(BaseMixinData):
    """
    Информация для загрузки исходников датасета
    """

    mode: Optional[SourceModeChoice]
    value: Optional[Union[HttpUrl, str]]
    path: Optional[DirectoryPath]

    @validator("value", allow_reuse=True)
    def _validate_mode_value(
        cls, value: Union[HttpUrl, str], values
    ) -> Union[HttpUrl, str]:
        mode = values.get("mode")
        if mode == SourceModeChoice.GoogleDrive:
            if not isinstance(value, str):
                raise ValueTypeException(value, str)
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


# class ColumnsProcessingData(BaseMixinData):
#     type: ColumnProcessingTypeChoice
#     parameters: Any
#
#     @validator("type", pre=True)
#     def _validate_type(
#         cls, value: ColumnProcessingTypeChoice
#     ) -> ColumnProcessingTypeChoice:
#         if not value:
#             return value
#         name = (
#             value
#             if isinstance(value, ColumnProcessingTypeChoice)
#             else ColumnProcessingTypeChoice(value)
#         ).name
#         cls.__fields__["parameters"].type_ = getattr(
#             column_processing, column_processing.ColumnProcessing[name].value
#         )
#         return value
#
#     @validator("parameters", always=True)
#     def _validate_parameters(cls, value: Any, values, field) -> Any:
#         return field.type_(**(value or {}))


class LayerBindData(BaseMixinData):
    up: List[PositiveInt] = []
    down: List[PositiveInt] = []


class CreationBlockData(IDMixinData):
    name: str
    type: LayerTypeChoice
    removable: bool = False
    bind: LayerBindData
    position: Tuple[int, int]
    parameters: Any

    def __init__(self, **data):
        data.update({"parameters": {**data.get("parameters", {})}})
        super().__init__(**data)

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerTypeChoice) -> LayerTypeChoice:
        if value not in list(LayerTypeChoice):
            raise EnumMemberError(enum_values=list(LayerTypeChoice))
        name = (
            value if isinstance(value, LayerTypeChoice) else LayerTypeChoice(value)
        ).name
        type_ = getattr(creations.blocks.types, f"Block{name.title()}Parameters")
        cls.__fields__["parameters"].required = True
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class CreationBlockList(UniqueListMixin):
    class Meta:
        source = CreationBlockData
        identifier = "id"


class DatasetCreationArchitectureData(BaseMixinData):
    inputs: CreationBlockList = CreationBlockList()
    outputs: CreationBlockList = CreationBlockList()


class CreationValidateBlocksData(BaseMixinData):
    type: LayerGroupChoice
    architecture: ArchitectureChoice
    items: DatasetCreationArchitectureData


class CreationVersionData(DatasetCreationArchitectureData):
    """
    Полная информация о создании версии датасета
    Inputs:
        alias: str - alias версии
        name: str - название версии
        datasets_path: pathlib.Path - путь к директории датасетов проекта (./TerraAI/datasets)
        info: CreationInfoData - соотношение обучающей выборки к валидационной, а также его перемешивание
        use_generator: bool - использовать генератор при обучении. По умолчанию: False
        inputs: CreationInputsList - входные слои
        outputs: CreationOutputsList - выходные слои
    """

    alias: Optional[AliasType]
    name: Optional[str]
    info: CreationInfoData = CreationInfoData()  # Train/Val split, shuffle

    _path: Path = PrivateAttr()

    def __init__(self, **data):
        if not data:
            data = {}
        if data.get("name"):
            data.update(
                {
                    "alias": data.get(
                        "alias",
                        re.sub(
                            r"([\-]+)",
                            "_",
                            slugify(data.get("name", ""), language_code="ru"),
                        ),
                    )
                }
            )
        self._path = Path(
            data.get("path"),
            "versions",
            f'{data.get("alias")}.{terra_ai_settings.DATASET_VERSION_EXT}',
        )
        super().__init__(**data)

    @property
    def path(self) -> Path:
        return self._path


class CreationData(AliasMixinData):
    """
    Полная информация о создании датасета
    """

    name: Optional[str]
    source: SourceData
    architecture: Optional[ArchitectureChoice]
    tags: List[str] = []
    first_creation: bool = True
    stage: PositiveInt = 1
    version: Optional[CreationVersionData]  # Optional больше сделано для дебаггинга

    _path: Path = PrivateAttr()

    def __init__(self, **data):
        if data.get("name"):
            data.update(
                {
                    "alias": data.get(
                        "alias",
                        re.sub(
                            r"([\-]+)",
                            "_",
                            slugify(data.get("name", ""), language_code="ru"),
                        ),
                    )
                }
            )
        self._path = Path(
            terra_ai_settings.TERRA_PATH.datasets,
            f'{data.get("alias")}.{terra_ai_settings.DATASET_EXT}',
        )
        os.makedirs(self._path, exist_ok=True)
        if data.get("version"):
            data.get("version").update({"path": self._path})
        super().__init__(**data)

    @property
    def path(self) -> Path:
        return self._path
