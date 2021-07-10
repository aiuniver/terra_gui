"""
## Структура данных слоев
"""

import json

from typing import Optional, List, Tuple, Any, Union
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import ConstrainedIntValueGe0
from ..exceptions import XYException
from .extra import LayerTypeChoice, LayerGroupChoice
from . import parameters


class LayerShapeData(BaseMixinData):
    """
    Размерности слоя
    """

    input: List[Tuple[PositiveInt, ...]] = []
    output: List[Tuple[PositiveInt, ...]] = []


class LayerBindData(BaseMixinData):
    """
    Связи слоев сверху и снизу
    """

    up: List[Optional[ConstrainedIntValueGe0]] = []
    down: List[ConstrainedIntValueGe0] = []

    @validator("up", allow_reuse=True)
    def _validate_bind(cls, value):
        if not value:
            return value
        if None in value:
            value = list(filter(None, value))
            value.insert(0, None)
        return value


class LayerData(AliasMixinData):
    """
    Данные слоя
    """

    name: str
    "Название"
    type: LayerTypeChoice
    "Тип слоя"
    group: LayerGroupChoice
    "Группа слоя"
    bind: LayerBindData = LayerBindData()
    "Связи со слоями"
    shape: LayerShapeData = LayerShapeData()
    "Размерности слоя"
    location: Optional[Tuple[ConstrainedIntValueGe0, ...]]
    "Расположение слоя в сетке модели"
    position: Optional[Tuple[int, ...]]
    "Расположение слоя в сетке модели"
    parameters: Optional[Any]
    "Параметры слоя"

    @property
    def parameters_dict(self) -> dict:
        __data = json.loads(self.parameters.main.json())
        __data.update(json.loads(self.parameters.extra.json()))
        return __data

    @validator("location", "position", allow_reuse=True)
    def _validate_xy(cls, value: list, values) -> list:
        if value is None:
            return value
        if len(value) != 2:
            raise XYException(values.get("alias"), value)
        return value

    @validator("type", allow_reuse=True, pre=True)
    def _validate_type(cls, value: LayerTypeChoice) -> LayerTypeChoice:
        if not hasattr(LayerTypeChoice, value):
            raise EnumMemberError(enum_values=list(LayerTypeChoice))
        type_ = getattr(parameters, getattr(parameters.ParametersType, value))
        cls.__fields__["parameters"].type_ = type_
        cls.__fields__["parameters"].required = True
        return value

    @validator("parameters", allow_reuse=True)
    def _validate_parameters(
        cls, value: Any, **kwargs
    ) -> Union[parameters.ParametersTypeUnion]:
        return kwargs.get("field").type_(**value)

    @validator("bind", allow_reuse=True)
    def _validate_bind(cls, value: LayerBindData, values) -> LayerBindData:
        if values.get("group") == LayerGroupChoice.input:
            value.up.insert(0, None)
        return value


class LayersList(UniqueListMixin):
    """
    Список слоев, основанных на `LayerData`
    ```
    class Meta:
        source = LayerData
        identifier = "alias"
    ```
    """

    class Meta:
        source = LayerData
        identifier = "alias"
