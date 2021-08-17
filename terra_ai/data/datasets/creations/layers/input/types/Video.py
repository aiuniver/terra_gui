from enum import Enum
from typing import Optional, List

from pydantic import validator
from pydantic.types import PositiveInt

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice


class FrameModeChoice(str, Enum):
    stretch = "Растянуть"
    keep_proportions = "Сохранить пропорции"


class FillModeChoice(str, Enum):
    black_frames = "Черными кадрами"
    average_value = "Средним значением"
    last_frames = "Последними кадрами"


class VideoModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    width: PositiveInt
    height: PositiveInt
    frame_mode: FrameModeChoice = FrameModeChoice.keep_proportions
    fill_mode: FillModeChoice = FillModeChoice.black_frames
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    video_mode: VideoModeChoice = VideoModeChoice.completely
    max_frames: Optional[PositiveInt]
    step: Optional[PositiveInt]
    length: Optional[PositiveInt]
    deploy: Optional[bool] = False

    @validator("video_mode", allow_reuse=True)
    def _validate_prepare_method(cls, value: VideoModeChoice) -> VideoModeChoice:
        if value == VideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        else:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
