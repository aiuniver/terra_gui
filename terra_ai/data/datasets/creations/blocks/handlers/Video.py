from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.datasets.extra import (
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
)
from terra_ai.data.datasets.creations.blocks.extra import VideoScalerData


class OptionsData(VideoScalerData):
    width: PositiveInt
    height: PositiveInt
    fill_mode: LayerVideoFillModeChoice
    frame_mode: LayerVideoFrameModeChoice
    video_mode: LayerVideoModeChoice
    max_frames: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]

    # Внутренние параметры
    deploy: Optional[bool] = False

    @validator("video_mode", allow_reuse=True)
    def _validate_video_mode(cls, value):
        if value == LayerVideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        elif value == LayerVideoModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
