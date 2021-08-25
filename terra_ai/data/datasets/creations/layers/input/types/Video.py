from typing import Optional, List

from pydantic import validator
from pydantic.types import PositiveInt

from ...extra import MinMaxScalerData, SourcesPathsData
from .....extra import (
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerScalerVideoChoice,
    LayerVideoModeChoice,
)


class ParametersData(MinMaxScalerData, SourcesPathsData):
    width: PositiveInt
    height: PositiveInt
    fill_mode: LayerVideoFillModeChoice = LayerVideoFillModeChoice.black_frames
    frame_mode: LayerVideoFrameModeChoice = LayerVideoFrameModeChoice.keep_proportions
    video_mode: LayerVideoModeChoice
    max_frames: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    scaler: LayerScalerVideoChoice

    cols_names: Optional[List[str]]
    deploy: Optional[bool] = False

    @validator("video_mode", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerVideoModeChoice
    ) -> LayerVideoModeChoice:
        if value == LayerVideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        elif value == LayerVideoModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
