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
        required_completely = value == LayerVideoModeChoice.completely
        cls.__fields__["max_frames"].required = required_completely
        cls.__fields__["length"].required = not required_completely
        cls.__fields__["step"].required = not required_completely
        return value
