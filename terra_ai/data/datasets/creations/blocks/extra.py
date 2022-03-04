from typing import Optional
from pydantic import validator

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerScalerDefaultChoice,
    LayerScalerImageChoice,
    LayerScalerVideoChoice,
)


class BaseOptionsData(BaseMixinData):
    def __init__(self, **data):
        items = {}
        for _key, _item in data.items():
            if (isinstance(_item, str) and _item == "") or _item is None:
                continue
            items[_key] = _item
        super().__init__(**items)


class MinMaxScalerData(BaseOptionsData):
    scaler: LayerScalerDefaultChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(cls, value):
        required = value == LayerScalerDefaultChoice.min_max_scaler
        cls.__fields__["min_scaler"].required = required
        cls.__fields__["max_scaler"].required = required
        return value


class ImageScalerData(BaseOptionsData):
    scaler: LayerScalerImageChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(cls, value):
        required = value in (
            LayerScalerImageChoice.min_max_scaler,
            LayerScalerImageChoice.terra_image_scaler,
        )
        cls.__fields__["min_scaler"].required = required
        cls.__fields__["max_scaler"].required = required
        return value


class VideoScalerData(BaseOptionsData):
    scaler: LayerScalerVideoChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(cls, value):
        required = value == LayerScalerVideoChoice.min_max_scaler
        cls.__fields__["min_scaler"].required = required
        cls.__fields__["max_scaler"].required = required
        return value
