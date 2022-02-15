from typing import Optional

from pydantic import validator
from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from ...extra import MinMaxScalerData
from .....extra import (
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerScalerVideoChoice,
    LayerVideoModeChoice,
)


class ParametersData(BaseMixinData, MinMaxScalerData):
    width: PositiveInt
    height: PositiveInt
    fill_mode: LayerVideoFillModeChoice = LayerVideoFillModeChoice.average_value
    frame_mode: LayerVideoFrameModeChoice = LayerVideoFrameModeChoice.fit
    video_mode: LayerVideoModeChoice
    max_frames: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    scaler: LayerScalerVideoChoice
    put: Optional[PositiveInt]

    deploy: Optional[bool] = False

    @validator("video_mode", allow_reuse=True)
    def _validate_video_mode(cls, value: LayerVideoModeChoice) -> LayerVideoModeChoice:
        if value == LayerVideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        elif value == LayerVideoModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
