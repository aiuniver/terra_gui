"""
## Структура данных моделей
"""

from typing import Optional, List, Dict
from pydantic import validator

from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..typing import confilepath, AliasType
from ..exceptions import (
    LayersNotInAvailableException,
    LayersNotDefinedException,
    PositionXYException,
)
from ..typing import Base64Type
from .layers import LayersList


class ModelDetailsData(BaseMixinData):
    """
    Детальная информация о модели
    """

    type: str
    "Тип модели: `2D`"
    name: AliasType
    "Название модели: `rasposnavanie_avtomobiley`"
    input_shape: str
    "Размерность входных слоев: `[32,32,3], [128,128,3]`"
    image: Base64Type
    "Изображение схемы модели в `base64`"


class ModelData(AliasMixinData):
    """
    Информация о модели
    """

    name: str
    "Название"
    file_path: Optional[confilepath(ext="model")]
    "Путь к файлу модели"
    details: Optional[ModelDetailsData]
    "Детальная информация о модели"
    layers: LayersList = LayersList()
    "Список слоев"
    position: Optional[Dict[str, List[int]]]
    "Позиция слоев в пользовательском интерфейсе, желательно указывать в процентах"
    location: Optional[List[List[Optional[str]]]]
    "Расположение слоев в пользовательском интерфейсе. Список должен состоять из LayerData.alias"

    @validator("position", allow_reuse=True)
    def _validate_position(cls, value: dict, values) -> dict:
        if value is None:
            return value
        __available = list(map(lambda item: item.alias, values.get("layers")))
        __layers = list(value.keys())

        __not_in_available = list(filter(None, list(set(__layers) - set(__available))))
        if len(__not_in_available):
            raise LayersNotInAvailableException(__not_in_available, __available)

        __not_defined_location = list(
            filter(None, list(set(__available) - set(__layers)))
        )
        if len(__not_defined_location):
            raise LayersNotDefinedException(__not_defined_location)

        for __layer, __value in value.items():
            if len(__value) != 2:
                raise PositionXYException(__layer, __value)

        return value

    @validator("location", allow_reuse=True)
    def _validate_location(cls, value: list, values) -> list:
        if value is None:
            return value
        __available = list(map(lambda item: item.alias, values.get("layers")))
        __layers = []
        for __group in value:
            __layers += __group

        __not_in_available = list(filter(None, list(set(__layers) - set(__available))))
        if len(__not_in_available):
            raise LayersNotInAvailableException(__not_in_available, __available)

        __not_defined_location = list(
            filter(None, list(set(__available) - set(__layers)))
        )
        if len(__not_defined_location):
            raise LayersNotDefinedException(__not_defined_location)

        return value


class ModelsList(UniqueListMixin):
    """
    Список моделей, основанных на `ModelData`
    ```
    class Meta:
        source = ModelData
        identifier = "alias"
    ```
    """

    class Meta:
        source = ModelData
        identifier = "alias"


class ModelsGroupData(AliasMixinData):
    """
    Группа моделей
    """

    name: str
    models: ModelsList = ModelsList()


class ModelsGroupsList(UniqueListMixin):
    """
    Список групп моделей, основанных на `ModelsGroupData`
    ```
    class Meta:
        source = ModelsGroupData
        identifier = "alias"
    ```
    """

    class Meta:
        source = ModelsGroupData
        identifier = "alias"
