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
    type: Optional[BlockFunctionTypeChoice]
    change_type: Optional[ChangeTypeAvailableChoice] = ChangeTypeAvailableChoice.int
    shape: Optional[List[PositiveInt]]
    min_scale: Optional[confloat(ge=0, le=1)] = 0
    max_scale: Optional[confloat(ge=0, le=1)] = 1
    class_id: Optional[conint(ge=0)] = 0
    classes_colors: Optional[List[Color]]
    open_tag: Optional[List[str]]
    close_tag: Optional[List[str]]
    alpha: Optional[confloat(gt=0, le=1)] = 0.5
    score_threshold: Optional[confloat(gt=0, le=1)] = 0.3
    iou_threshold: Optional[confloat(gt=0, le=1)] = 0.45
    method: Optional[
        PostprocessBoxesMethodAvailableChoice
    ] = PostprocessBoxesMethodAvailableChoice.nms
    sigma: Optional[confloat(gt=0, le=1)] = 0.3
    classes: Optional[List[str]]
    colors: Optional[List[Color]]
    line_thickness: Optional[PositiveInt]

    def __init__(self, **data):
        _type = data.get("type")
        _keys = ["group", "type"]
        if _type == BlockFunctionTypeChoice.ChangeType:
            _keys += ["change_type"]
        elif _type == BlockFunctionTypeChoice.ChangeSize:
            _keys += ["shape"]
        elif _type == BlockFunctionTypeChoice.MinMaxScale:
            _keys += ["min_scale", "max_scale"]
        elif _type == BlockFunctionTypeChoice.MaskedImage:
            _keys += ["class_id"]
        elif _type == BlockFunctionTypeChoice.PlotMaskSegmentation:
            _keys += ["classes_colors"]
        elif _type == BlockFunctionTypeChoice.PutTag:
            _keys += ["open_tag", "close_tag", "alpha"]
        elif _type == BlockFunctionTypeChoice.PostprocessBoxes:
            _keys += ["score_threshold", "iou_threshold", "method", "sigma"]
        elif _type == BlockFunctionTypeChoice.PlotBBoxes:
            _keys += ["classes", "colors", "line_thickness"]
        data = dict(filter(lambda item: item[0] in _keys, data.items()))
        super().__init__(**data)

    @validator("type", pre=True)
    def _validate_type(cls, value: BlockFunctionTypeChoice) -> BlockFunctionTypeChoice:
        for name, item in cls.__fields__.items():
            if name in ["group", "type"]:
                continue
            cls.__fields__[name].required = False
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
