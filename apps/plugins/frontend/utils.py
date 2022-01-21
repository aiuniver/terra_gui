from enum import Enum, EnumMeta
from typing import Any
from pydantic.types import ConstrainedNumberMeta

from terra_ai.data.modeling.layers.extra import (
    PaddingAddCausalChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    PaddingChoice,
    InterpolationChoice,
    ResizingInterpolationChoice,
    PretrainedModelWeightsChoice,
    PretrainedModelPoolingChoice,
    YOLOModeChoice,
    YOLOActivationChoice,
    VAELatentRegularizerChoice,
    SpaceToDepthDataFormatChoice,
    CONVBlockConfigChoice,
    ResblockActivationChoice,
    NormalizationChoice,
    MergeLayerChoice,
    ConditionalMergeModeChoice
)

from terra_ai.data.cascades.blocks.extra import (
    BlockOutputDataSaveAsChoice,
    BlockFunctionGroupChoice,
    BlockCustomGroupChoice
)

from .base import Field
from .extra import FieldTypeChoice

CHECKBOX_TYPES = [
    bool,
]
NUMBER_TYPES = [
    int,
    float,
    ConstrainedNumberMeta,
]
SELECT_TYPES = [
    PaddingAddCausalChoice,
    ActivationChoice,
    DataFormatChoice,
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
    PaddingChoice,
    InterpolationChoice,
    ResizingInterpolationChoice,
    PretrainedModelWeightsChoice,
    PretrainedModelPoolingChoice,
    YOLOModeChoice,
    YOLOActivationChoice,
    VAELatentRegularizerChoice,
    SpaceToDepthDataFormatChoice,
    CONVBlockConfigChoice,
    BlockOutputDataSaveAsChoice,
    BlockFunctionGroupChoice,
    BlockCustomGroupChoice,
    ResblockActivationChoice,
    NormalizationChoice,
    MergeLayerChoice,
    ConditionalMergeModeChoice
]


class Labels(str, Enum):
    block_size = "Размер блока"
    save_as = "Сохранить как"
    group = "Группа"
    type = "Выбор типа"
    max_age = "Количество кадров для остановки слежения"
    min_hits = "Количество кадров для возобновления отслеживания"
    postprocess = "Использовать постобработку"
    path = "Путь к обученной модели"
    shape = "Размерность"
    min_scale = "Минимальное значение"
    max_scale = "Максимальное значение"
    class_id = "ID класса"
    classes_colors = "Цвета классов"
    open_tag = "Открывающие тэги"
    close_tag = "Закрывающие тэги"
    alpha = "Альфа - порог вероятности"
    score_threshold = "Порог вероятности классов"
    iou_threshold = "Порог пересечения"
    method = "Метод подавления немаксимумов"
    sigma = "Коэффициент сглаживания"
    classes = "Имена классов"
    colors = "Цвета классов"
    line_thickness = "Толщина линии рамки"


class ChoiceValues(str, Enum):
    source = "Датасет source"
    file = "Файл"
    ChangeType = "Изменение типа данных"
    ChangeSize = "Изменение размера данных"
    MinMaxScale = "Нормализация (скелер)"
    MaskedImage = "Наложение маски по классу на изображение"
    PlotMaskSegmentation = "Наложение маски всех классов по цветам"
    PutTag = "Растановка тэгов по вероятностям из модели"
    PostprocessBoxes = "Постобработка yolo"
    PlotBboxes = "Наложение bbox на изображение"


def __prepare_label(value: str) -> str:
    items = list(filter(None, str(value).split("_"))) or [""]
    if len(items[0]):
        items[0] = f"{items[0][0].title()}{items[0][1:]}"
    return " ".join(items)


def __prepare_choice_value(value: Any) -> Any:
    if issubclass(value.__class__.__class__, EnumMeta):
        try:
            return ChoiceValues[value.value].value
        except KeyError:
            pass

    return value


def prepare_pydantic_field(field, parse: str) -> Field:
    __value = field.default
    __list = None

    if field.outer_type_.__class__ in NUMBER_TYPES or field.outer_type_ in NUMBER_TYPES:
        __type = FieldTypeChoice.number
    elif field.outer_type_ in CHECKBOX_TYPES:
        __type = FieldTypeChoice.checkbox
    elif field.outer_type_ in SELECT_TYPES:
        __type = FieldTypeChoice.select
        __list = list(
            map(
                lambda item: {"value": item, "label": __prepare_label(item)},
                field.outer_type_.values(),
            )
        )
        if field.allow_none:
            __list = [{"value": "__null__", "label": ""}] + __list
        if not __value:
            __value = field.default.name if field.default else None
    elif hasattr(field.outer_type_, "__origin__"):
        if field.outer_type_.__origin__ is tuple:
            __type = FieldTypeChoice.text_array
            __value = __value or None
        else:
            __type = FieldTypeChoice.text
            __value = str(__value)
    else:
        __type = FieldTypeChoice.text
        __value = None

    try:
        __label = Labels[field.name]
    except KeyError:
        __label = __prepare_label(field.name)

    return Field(
        type=__type,
        name=field.name,
        label=__label,
        parse=parse,
        value=__prepare_choice_value(field.default),
        list=__list,
    )
