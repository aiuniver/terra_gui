"""
## Структура данных для работы с датасетами
"""

from math import fsum
from typing import Union, Optional, Any
from pathlib import PosixPath
from pydantic import validator, FilePath, HttpUrl
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData, UniqueListMixin, AliasMixinData
from ..validators import validate_part_value
from ..extra import LayerInputTypeChoice, LayerOutputTypeChoice
from ..exceptions import (
    ZipFileException,
    ValueTypeException,
    PartTotalException,
    ListEmptyException,
)
from .extra import SourceModeChoice
from .tags import TagsListData
from . import parameters


class SourceData(BaseMixinData):
    """
    Информация для загрузки исходников датасета
    """

    mode: SourceModeChoice
    "Режим загрузки исходных данных"
    value: Union[FilePath, HttpUrl]
    "Значение для режим загрузки исходных данных. Тип будет зависеть от выбранного режима `mode`"

    @validator("value", allow_reuse=True)
    def _validate_mode_value(
        cls, value: Union[PosixPath, HttpUrl], **kwargs
    ) -> Union[PosixPath, HttpUrl]:
        if isinstance(value, PosixPath):
            split_value = str(value).split(".")
            if len(split_value) < 2 or split_value[-1].lower() != "zip":
                raise ZipFileException(value)

        mode = kwargs.get("values", {}).get("mode")

        if mode == SourceModeChoice.google_drive:
            if not isinstance(value, PosixPath):
                raise ValueTypeException(value, PosixPath)

        if mode == SourceModeChoice.url:
            if not isinstance(value, HttpUrl):
                raise ValueTypeException(value, HttpUrl)

        return value


class CreationInfoPartData(BaseMixinData):
    """
    Доли использования данных для обучающей, тестовой и валидационной выборок"
    """

    train: float = 0.6
    "Обучающая выборка"
    validation: float = 0.3
    "Валидационная выборка"
    test: float = 0.1
    "Тестовая выборка"

    _validate_values = validator("train", "validation", "test", allow_reuse=True)(
        validate_part_value
    )

    @property
    def total(self) -> float:
        """
        Сумма всех значений
        """
        return fsum([self.train, self.validation, self.test])


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


class CreationInputData(AliasMixinData):
    """
    Информация о `input`-слое
    """

    name: str
    "Название"
    type: LayerInputTypeChoice
    "Тип данных"
    parameters: Optional[Any]
    "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    @validator("type", allow_reuse=True, pre=True)
    def _validate_type(cls, value: LayerInputTypeChoice) -> LayerInputTypeChoice:
        if not hasattr(LayerInputTypeChoice, value):
            raise EnumMemberError(enum_values=list(LayerInputTypeChoice))
        type_ = getattr(parameters, getattr(parameters.LayerInputDatatype, value))
        cls.__fields__["parameters"].type_ = type_
        cls.__fields__["parameters"].required = True
        return value

    @validator("parameters", allow_reuse=True)
    def _validate_parameters(
        cls, value: Any, **kwargs
    ) -> Union[parameters.LayerInputDatatypeUnion]:
        return kwargs.get("field").type_(**value)


class CreationOutputData(AliasMixinData):
    """
    Информация о `output`-слое
    """

    name: str
    "Название"
    type: LayerOutputTypeChoice
    "Тип данных"
    parameters: Optional[Any]
    "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    @validator("type", allow_reuse=True, pre=True)
    def _validate_type(cls, value: LayerOutputTypeChoice) -> LayerOutputTypeChoice:
        if not hasattr(LayerOutputTypeChoice, value):
            raise EnumMemberError(enum_values=list(LayerOutputTypeChoice))
        type_ = getattr(parameters, getattr(parameters.LayerOutputDatatype, value))
        cls.__fields__["parameters"].type_ = type_
        cls.__fields__["parameters"].required = True
        return value

    @validator("parameters", allow_reuse=True)
    def _validate_parameters(
        cls, value: Any, **kwargs
    ) -> Union[parameters.LayerOutputDatatypeUnion]:
        return kwargs.get("field").type_(**value)


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
        identifier = "alias"


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
        identifier = "alias"


class CreationData(BaseMixinData):
    """
    Полная информация о создании датасета
    """

    name: str
    "Название"
    info: CreationInfoData = CreationInfoData()
    "Информация о данных"
    tags: TagsListData = TagsListData()
    "Список тегов"
    inputs: CreationInputsList = CreationInputsList()
    "`input`-слои"
    outputs: CreationOutputsList = CreationOutputsList()
    "`output`-слои"

    @validator("inputs", "outputs", allow_reuse=True)
    def _validate_required(cls, value: UniqueListMixin) -> UniqueListMixin:
        if not len(value):
            raise ListEmptyException(type(value))
        return value
