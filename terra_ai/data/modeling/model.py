"""
## Структура данных моделей
"""

from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt

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

    value: confilepath(ext="model")
    "Пусть к фалу модели"


class BlockDetailsData(BaseMixinData):
    """
    Детальная информация о блоке
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

    @property
    def links(self) -> list:
        links = LinksList()
        for layer in self.layers:
            links += self.__get_links(layer.id, layer.bind.down)
        return links

    def __get_links(
        self, layer_id: PositiveInt, binds_down: LayerBindData
    ) -> LinksList:
        links = LinksList()
        for bind_down in binds_down:
            if not bind_down:
                continue
            for bind_up in self.layers.get(bind_down[0]).bind.up:
                if bind_up[0] == layer_id:
                    links.append(
                        LinkData(
                            originID=bind_up[0],
                            originSlot=bind_down[1],
                            targetID=bind_down[0],
                            targetSlot=bind_up[1],
                        )
                    )
                    break
        return links

    def dict(self, **kwargs):
        data = super().dict()
        data.update({"links": self.links})
        return data


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
    references: BlocksList = BlocksList()
    "Списки блоков, используемых в модели"

    @property
    def links(self) -> list:
        links = LinksList()
        for layer in self.layers:
            links += self.__get_links(layer.id, layer.bind.down)
        return links

    def __get_links(
        self, layer_id: PositiveInt, binds_down: LayerBindData
    ) -> LinksList:
        links = LinksList()
        for bind_down in binds_down:
            if not bind_down:
                continue
            for bind_up in self.layers.get(bind_down[0]).bind.up:
                if bind_up[0] == layer_id:
                    links.append(
                        LinkData(
                            originID=bind_up[0],
                            originSlot=bind_down[1],
                            targetID=bind_down[0],
                            targetSlot=bind_up[1],
                        )
                    )
                    break
        return links

    def dict(self, **kwargs):
        data = super().dict()
        data.update({"links": self.links})
        return data


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
