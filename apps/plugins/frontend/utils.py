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
    SpaceToDepthDataFormatChoice,
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
]


class Labels(str, Enum):
    block_size = "Размер блока"


def __prepare_label(value: str) -> str:
    items = list(filter(None, str(value).split("_"))) or [""]
    if len(items[0]):
        items[0] = f"{items[0][0].title()}{items[0][1:]}"
    return " ".join(items)


def prepare_pydantic_field(field, parse: str) -> Field:
    __value = "" if field.default is None else field.default
    __list = None

    if field.type_.__class__ in NUMBER_TYPES or field.type_ in NUMBER_TYPES:
        __type = FieldTypeChoice.number
    elif field.type_ in CHECKBOX_TYPES:
        __type = FieldTypeChoice.checkbox
    elif field.type_ in SELECT_TYPES:
        __type = FieldTypeChoice.select
        __list = list(
            map(
                lambda item: {"value": item, "label": __prepare_label(item)},
                field.type_.values(),
            )
        )
        if not field.required:
            __list = [{"value": "__null__", "label": ""}] + __list
        if not __value:
            __value = __list[0].get("value")
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
