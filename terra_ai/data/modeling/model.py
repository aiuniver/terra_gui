"""
## Структура данных моделей
"""

from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt

from terra_ai.data.modeling.layer import LayersList
from ... import settings
from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, AliasType, Base64Type
from .extra import LayerBindPositionChoice, LayerGroupChoice


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


class BlockDetailsData(AliasMixinData):
    """
    Детальная информация о блоке
    """

    name: Optional[str]
    "Название блока"
    image: Optional[Base64Type]
    "Изображение схемы блока в `base64`"
    layers: Optional[LayersList]
    "Список слоев"
    keras: Optional[str] = ""
    "Код на keras"


class BlockData(AliasMixinData):
    """
    Информация о блоке
    """

    name: Optional[str]
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
        identifier = "alias"


class ModelDetailsData(AliasMixinData):
    """
    Детальная информация о модели
    """

    name: Optional[str]
    "Название модели: `Распознавание автомобилей`"
    image: Optional[Base64Type]
    "Изображение схемы модели в `base64`"
    layers: LayersList = []
    "Список слоев"
    references: BlocksList = BlocksList()
    "Списки блоков, используемых в модели"
    keras: Optional[str] = ""
    "Код на keras"

    #     def __str__(self):
    #         if self.image:
    #             _is_image = "\033[1;32mYes\033[0m"
    #         else:
    #             _is_image = "\033[1;31mNo\033[0m"
    #         if self.keras:
    #             _is_keras = "\033[1;32mYes\033[0m"
    #         else:
    #             _is_keras = "\033[1;31mNo\033[0m"
    #
    #         # name: str
    #         # "Название"
    #         # type: LayerTypeChoice
    #         # "Тип слоя"
    #         # group: LayerGroupChoice
    #         # "Группа слоя"
    #         # bind: LayerBindData = LayerBindData()
    #         # "Связи со слоями"
    #         # shape: LayerShapeData = LayerShapeData()
    #         # "Размерности слоя"
    #         # task: Optional[Union[LayerInputTypeChoice, LayerOutputTypeChoice]]
    #         # "Тип задачи"
    #         # num_classes: Optional[PositiveInt]
    #         # "Количество классов"
    #         # position: Optional[Tuple[int, int]]
    #         # "Расположение слоя в сетке модели"
    #         # parameters: Any
    #         # "Параметры слоя"
    #         # reference: Optional[ReferenceLayerType]
    #         # "Ссылка на блок, описанный в модели в поле `references`"
    #
    #         _layers = ""
    #         for index, layer in enumerate(self.layers):
    #             _layers += f"      {index+1} {layer.type} [{layer.name}]\n"
    #
    #         output = f"""—————————————————————————————————————————————————
    # Model \033[0;36m{self.name} [{self.alias}]\033[0m
    #       Image : {_is_image}
    #       Keras : {_is_keras}
    #       ———————————————————————————————————————————
    # {_layers}—————————————————————————————————————————————————"""
    #         return output

    @property
    def inputs(self) -> LayersList:
        layers = LayersList()
        for layer in self.layers:
            if layer.group == LayerGroupChoice.input:
                layers.append(layer)
        return layers

    @property
    def middles(self) -> LayersList:
        layers = LayersList()
        for layer in self.layers:
            if layer.group == LayerGroupChoice.middle:
                layers.append(layer)
        return layers

    @property
    def outputs(self) -> LayersList:
        layers = LayersList()
        for layer in self.layers:
            if layer.group == LayerGroupChoice.output:
                layers.append(layer)
        return layers


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
