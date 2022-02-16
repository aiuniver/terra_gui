from typing import Optional
from pydantic import validator

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerScalerDefaultChoice,
    LayerScalerImageChoice,
    LayerScalerVideoChoice,
)


class MinMaxScalerData(BaseMixinData):
    scaler: LayerScalerDefaultChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(
        cls, value: LayerScalerDefaultChoice
    ) -> LayerScalerDefaultChoice:
        if value == LayerScalerDefaultChoice.min_max_scaler:
            cls.__fields__["min_scaler"].required = True
            cls.__fields__["max_scaler"].required = True
        return value


class ImageScalerData(BaseMixinData):
    scaler: LayerScalerImageChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(cls, value: LayerScalerImageChoice) -> LayerScalerImageChoice:
        if value in (
            LayerScalerImageChoice.min_max_scaler,
            LayerScalerImageChoice.terra_image_scaler,
        ):
            cls.__fields__["min_scaler"].required = True
            cls.__fields__["max_scaler"].required = True
        return value


class VideoScalerData(BaseMixinData):
    scaler: LayerScalerVideoChoice
    min_scaler: Optional[int] = 0
    max_scaler: Optional[int] = 1

    @validator("scaler")
    def _validate_scaler(cls, value: LayerScalerVideoChoice) -> LayerScalerVideoChoice:
        if value == LayerScalerVideoChoice.min_max_scaler:
            cls.__fields__["min_scaler"].required = True
            cls.__fields__["max_scaler"].required = True
        return value
