from enum import Enum
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
    SpaceToDepthDataFormatChoice, CONVBlockConfigChoice,
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
]


class Labels(str, Enum):
    block_size = "Размер блока"
    save_as = "Сохранить как"
    group = "Группа"
    postprocess = "Использовать постобработку"


def __prepare_label(value: str) -> str:
    items = list(filter(None, str(value).split("_"))) or [""]
    if len(items[0]):
        items[0] = f"{items[0][0].title()}{items[0][1:]}"
    return " ".join(items)


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
    else:
        if field.outer_type_.__origin__ is tuple:
            __type = FieldTypeChoice.text_array
            __value = __value or None
        else:
            __type = FieldTypeChoice.text
            __value = str(__value)

    try:
        __label = Labels[field.name]
    except KeyError:
        __label = __prepare_label(field.name)

    return Field(
        type=__type,
        name=field.name,
        label=__label,
        parse=parse,
        value=__value,
        list=__list,
    )
