"""
## Структура данных моделей
"""

from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt

from ... import settings
from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, AliasType, Base64Type
from .layer import LayersList, LayerBindData
from .extra import LayerBindPositionChoice


class LinkData(BaseMixinData):
    originID: PositiveInt
    originSlot: LayerBindPositionChoice
    targetID: PositiveInt
    targetSlot: LayerBindPositionChoice


class LinksList(List[LinkData]):
    def dict(self) -> list:
        output = []
        for link in self:
            output.append(link.dict())
        return output


class ModelLoadData(BaseMixinData):
    """
    Информация для загрузки модели
    """

    value: confilepath(ext=settings.MODEL_EXT)
    "Пусть к фалу модели"


class BlockDetailsData(BaseMixinData):
    """
    Детальная информация о блоке
    """

    name: Optional[AliasType]
    "Название модели: `rasposnavanie_avtomobiley`"
    image: Optional[Base64Type]
    "Изображение схемы модели в `base64`"
    layers: Optional[LayersList]
    "Список слоев"


class BlockData(AliasMixinData):
    """
    Информация о блоке
    """

    name: str
    "Название"
    file_path: Optional[confilepath(ext="block")]
    "Путь к файлу блока"
    details: Optional[BlockDetailsData]
    "Детальная информация о блоке"


class BlocksList(UniqueListMixin):
    """
    Список блоков, основанных на `BlockData`
    ```
    class Meta:
        source = BlockData
        identifier = "name"
    ```
    """

    class Meta:
        source = BlockData
        identifier = "name"


class ModelDetailsData(AliasMixinData):
    """
    Детальная информация о модели
    """

    name: Optional[str]
    "Название модели: `Распознавание автомобилей`"
    image: Optional[Base64Type]
    "Изображение схемы модели в `base64`"
    layers: Optional[LayersList]
    "Список слоев"
    references: BlocksList = BlocksList()
    "Списки блоков, используемых в модели"


class ModelData(AliasMixinData):
    """
    Информация о модели
    """

    name: str
    "Название"
    file_path: Optional[confilepath(ext=settings.MODEL_EXT)]
    "Путь к файлу модели"
    details: Optional[ModelDetailsData]
    "Детальная информация о модели"


class ModelListData(BaseMixinData):
    """
    Информация о модели в списке
    """

    value: confilepath(ext=settings.MODEL_EXT)
    "Путь к файлу модели"
    label: Optional[str]
    "Название"

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = values.get("value")
        if not file_path:
            return value
        return file_path.name.split(f".{settings.MODEL_EXT}")[0]


class ModelsList(UniqueListMixin):
    """
    Список моделей, основанных на `ModelListData`
    ```
    class Meta:
        source = ModelListData
        identifier = "label"
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
