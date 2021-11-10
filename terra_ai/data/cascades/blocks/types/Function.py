from typing import List, Optional
from pydantic.color import Color
from pydantic import validator
from pydantic.types import PositiveInt, conint, confloat

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import (
    BlockFunctionGroupChoice,
    BlockFunctionTypeChoice,
    ChangeTypeAvailableChoice,
    PostprocessBoxesMethodAvailableChoice,
)


class ParametersMainData(BaseMixinData):
    group: BlockFunctionGroupChoice
    type: BlockFunctionTypeChoice
    change_type: Optional[ChangeTypeAvailableChoice] = ChangeTypeAvailableChoice.int
    shape: Optional[List[int]]
    min_scale: Optional[float] = 0
    max_scale: Optional[float] = 1
    class_id: Optional[conint(ge=0)] = 0
    classes_colors: Optional[List[Color]]
    open_tag: Optional[List[str]]
    close_tag: Optional[List[str]]
    alpha: Optional[confloat(gt=0, le=1)] = 0.5
    score_threshold: Optional[confloat(gt=0, le=1)] = 0.3
    iou_threshold: Optional[confloat(gt=0, le=1)] = 0.45
    method: Optional[PostprocessBoxesMethodAvailableChoice] = PostprocessBoxesMethodAvailableChoice.nms
    sigma: Optional[confloat(gt=0, le=1)] = 0.3
    classes: Optional[List[str]]
    colors: Optional[List[Color]]
    line_thickness: Optional[PositiveInt]


    @validator("type")
    def _validate_type(cls, value: BlockFunctionTypeChoice) -> BlockFunctionTypeChoice:
        if value == BlockFunctionTypeChoice.ChangeType:
            cls.__fields__["change_type"].required = True
        elif value == BlockFunctionTypeChoice.ChangeSize:
            cls.__fields__["shape"].required = True
        elif value == BlockFunctionTypeChoice.MinMaxScale:
            cls.__fields__["min_scale"].required = True
            cls.__fields__["max_scale"].required = True
        elif value == BlockFunctionTypeChoice.MaskedImage:
            cls.__fields__["class_id"].required = True
        elif value == BlockFunctionTypeChoice.PlotMaskSegmentation:
            cls.__fields__["classes_colors"].required = True
        elif value == BlockFunctionTypeChoice.PutTag:
            cls.__fields__["open_tag"].required = True
            cls.__fields__["close_tag"].required = True
            cls.__fields__["alpha"].required = True
        elif value == BlockFunctionTypeChoice.PostprocessBoxes:
            cls.__fields__["score_threshold"].required = True
            cls.__fields__["iou_threshold"].required = True
            cls.__fields__["method"].required = True
            cls.__fields__["sigma"].required = True
        elif value == BlockFunctionTypeChoice.PlotBBoxes:
            cls.__fields__["classes"].required = True
            cls.__fields__["colors"].required = True
            cls.__fields__["line_thickness"].required = True
     
        return value
