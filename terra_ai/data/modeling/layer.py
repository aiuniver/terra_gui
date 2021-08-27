"""
## Структура данных слоев
"""

from typing import Optional, List, Tuple, Any
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from ..mixins import BaseMixinData, IDMixinData, UniqueListMixin
from ..exceptions import XYException
from . import layers
from .extra import LayerTypeChoice, LayerGroupChoice
from .types import ReferenceLayerType


class LayerShapeData(BaseMixinData):
    """
    Размерности слоя
    """

    input: List[Tuple[PositiveInt, ...]] = []
    output: List[Tuple[PositiveInt, ...]] = []


class LayerBindIDsData(BaseMixinData):
    """
    Связи слоев сверху и снизу, только ID
    """

    up: List[Optional[PositiveInt]] = []
    down: List[PositiveInt] = []


class LayerBindData(BaseMixinData):
    """
    Связи слоев сверху и снизу
    """

    up: List[Optional[PositiveInt]] = []
    down: List[PositiveInt] = []


class LayerData(IDMixinData):
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
    position: Optional[Tuple[int, int]]
    "Расположение слоя в сетке модели"
    parameters: Any
    "Параметры слоя"
    reference: Optional[ReferenceLayerType]
    "Ссылка на блок, описанный в модели в поле `references`"

    @property
    def bind_ids(self) -> LayerBindIDsData:
        return LayerBindIDsData(
            up=list(
                map(lambda item: item[0] if item is not None else None, self.bind.up)
            ),
            down=list(
                map(lambda item: item[0] if item is not None else None, self.bind.down)
            ),
        )

    @validator("bind", always=True)
    def _validate_bind(cls, value: LayerBindData, values) -> LayerBindData:
        if values.get("group") == LayerGroupChoice.input:
            value.up = list(filter(None, value.up))
            value.up.insert(0, None)
        return value

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerTypeChoice) -> LayerTypeChoice:
        if value not in list(LayerTypeChoice):
            raise EnumMemberError(enum_values=list(LayerTypeChoice))
        name = (
            value if isinstance(value, LayerTypeChoice) else LayerTypeChoice(value)
        ).name
        type_ = getattr(layers, getattr(layers.Layer, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class LayersList(UniqueListMixin):
    """
    Список слоев, основанных на `LayerData`
    ```
    class Meta:
        source = LayerData
        identifier = "id"
    ```
    """

    class Meta:
        source = LayerData
        identifier = "id"
