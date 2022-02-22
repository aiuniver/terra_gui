"""
## Структура данных моделей
"""

from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt
from dict_recursive_update import recursive_update

from terra_ai.data.modeling.layer import LayersList
from ... import settings
from ..mixins import BaseMixinData, AliasMixinData, UniqueListMixin
from ..types import confilepath, Base64Type
from .extra import LayerBindPositionChoice, LayerGroupChoice
from .layer import LayerData


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


class ModelBaseDetailsData(AliasMixinData):
    """
    Базова информация о проектировании
    """

    name: Optional[str]
    "Название модели: `Распознавание автомобилей`"
    image: Optional[Base64Type]
    "Изображение схемы модели в `base64`"
    layers: LayersList = []
    "Список слоев"
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

    @property
    def input_shape(self) -> str:
        shapes = []
        for layer in self.inputs:
            shapes += layer.shape.input
        return str(shapes)

    def reindex(self, source_id: PositiveInt, target_id: PositiveInt):
        layer_source = self.layers.get(source_id)
        layer_target = self.layers.get(target_id)
        layer_target.id = max(self.layers.ids) + 1
        layer_source.id = target_id
        layer_target.id = source_id

    def switch_index(
        self, source_id: PositiveInt, target_id: PositiveInt
    ) -> PositiveInt:
        if source_id == target_id or not len(self.layers):
            return
        layer_target = self.layers.get(target_id)
        _id_intermediate = None
        if layer_target:
            _id_intermediate = self.switch_index(target_id, max(self.layers.ids) + 1)
        layer_source = self.layers.get(source_id)
        for _id in layer_source.bind.up:
            if _id is None:
                continue
            _binds = self.layers.get(_id).bind.down
            if layer_source.id not in _binds:
                continue
            _binds[_binds.index(layer_source.id)] = target_id
            self.layers.get(_id).bind.down = _binds
        for _id in layer_source.bind.down:
            if _id is None:
                continue
            _binds = self.layers.get(_id).bind.up
            if layer_source.id not in _binds:
                continue
            _binds[_binds.index(layer_source.id)] = target_id
            self.layers.get(_id).bind.up = _binds
        layer_source.id = target_id
        if _id_intermediate:
            self.switch_index(_id_intermediate, source_id)
        return target_id

    def set_dataset_indexes(self, dataset):
        dataset_model = dataset.model

        for _index, _dataset_layer in enumerate(dataset_model.inputs):
            self.switch_index(self.inputs[_index].id, _dataset_layer.id)
        for _index, _dataset_layer in enumerate(dataset_model.outputs):
            self.switch_index(self.outputs[_index].id, _dataset_layer.id)

    def update_layers(self, dataset, exclude_type=False):
        print(dataset)
        dataset_model = dataset.model

        for index, layer in enumerate(self.inputs):
            dataset_layer = dataset_model.inputs.get(layer.id).native()
            dataset_layer.pop("id")
            dataset_layer.pop("bind")
            dataset_layer.pop("position")
            dataset_layer.pop("parameters")
            if exclude_type:
                dataset_layer.pop("type")
            self.layers.append(
                LayerData(**recursive_update(layer.native(), dataset_layer))
            )

        for index, layer in enumerate(self.outputs):
            dataset_layer = dataset_model.outputs.get(layer.id).native()
            dataset_layer.pop("id")
            dataset_layer.pop("bind")
            dataset_layer.pop("position")
            dataset_layer.pop("parameters")
            if exclude_type:
                dataset_layer.pop("type")
            self.layers.append(
                LayerData(**recursive_update(layer.native(), dataset_layer))
            )

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        data.update({"input_shape": self.input_shape})
        return data


class BlockDetailsData(ModelBaseDetailsData):
    """
    Детальная информация о блоке
    """

    pass


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


class ModelDetailsData(ModelBaseDetailsData):
    """
    Детальная информация о модели
    """

    references: BlocksList = BlocksList()


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
