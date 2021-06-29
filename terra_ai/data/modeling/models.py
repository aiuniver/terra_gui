"""
## Структура данных моделей
"""

from typing import Optional

from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..typing import confilepath, AliasType, Base64Type
from .layers import LayersList


class ModelDetailsData(BaseMixinData):
    """
    Детальная информация о модели
    """

    type: Optional[str]
    "Тип модели: `2D`"
    name: Optional[AliasType]
    "Название модели: `rasposnavanie_avtomobiley`"
    input_shape: Optional[str]
    "Размерность входных слоев: `[32,32,3], [128,128,3]`"
    image: Optional[Base64Type]
    "Изображение схемы модели в `base64`"


class ModelData(AliasMixinData):
    """
    Информация о модели
    """

    name: str
    "Название"
    file_path: Optional[confilepath(ext="model")]
    "Путь к файлу модели"
    details: ModelDetailsData = ModelDetailsData()
    "Детальная информация о модели"
    layers: LayersList = LayersList()
    "Список слоев"


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
