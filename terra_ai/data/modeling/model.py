"""
## Структура данных моделей
"""

from typing import Optional
from pydantic import validator

from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, AliasType, Base64Type
from .layer import LayersList


class ModelLoadData(BaseMixinData):
    """
    Информация для загрузки модели
    """

    value: confilepath(ext="model")
    "Пусть к фалу модели"


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
    layers: Optional[LayersList]
    "Список слоев"


class ModelData(AliasMixinData):
    """
    Информация о модели
    """

    name: str
    "Название"
    file_path: confilepath(ext="model")
    "Путь к файлу модели"
    details: Optional[ModelDetailsData]
    "Детальная информация о модели"


class ModelListData(BaseMixinData):
    """
    Информация о модели в списке
    """

    value: confilepath(ext="model")
    "Путь к файлу модели"
    label: Optional[str]
    "Название"

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = values.get("value")
        if not file_path:
            return value
        return file_path.name.split(".model")[0]


class ModelsList(UniqueListMixin):
    """
    Список моделей, основанных на `ModelListData`
    ```
    class Meta:
        source = ModelListData
        identifier = "name"
    ```
    """

    class Meta:
        source = ModelListData
        identifier = "label"


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
