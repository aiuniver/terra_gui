"""
## Структура данных слоев
"""

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

    input: Tuple[PositiveInt, ...] = ()
    output: Tuple[PositiveInt, ...] = ()


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
    bind: List[int] = []
    "Связи со слоями"
    shape: LayerShapeData = LayerShapeData()
    "Размерности слоя"
    location: Optional[List[ConstrainedIntValueGe0]]
    "Расположение слоя в сетке модели"
    position: Optional[List[int]]
    "Расположение слоя в сетке модели"
    parameters: Optional[Any]
    "Параметры слоя"

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
