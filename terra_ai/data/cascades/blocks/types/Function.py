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
    ObjectDetectionFilterClassesList,
)


class ParametersMainData(BaseMixinData):
    group: BlockFunctionGroupChoice = BlockFunctionGroupChoice.ObjectDetection
    type: Optional[BlockFunctionTypeChoice] = BlockFunctionTypeChoice.PostprocessBoxes
    change_type: Optional[ChangeTypeAvailableChoice] = ChangeTypeAvailableChoice.int
    shape: Optional[List[PositiveInt]]
    min_scale: Optional[confloat(ge=0, le=1)] = 0
    max_scale: Optional[confloat(ge=0, le=1)] = 1
    alpha: Optional[confloat(gt=0, le=1)] = 0.5
    score_threshold: Optional[confloat(gt=0, le=1)] = 0.3
    iou_threshold: Optional[confloat(gt=0, le=1)] = 0.45
    method: Optional[
        PostprocessBoxesMethodAvailableChoice
    ] = PostprocessBoxesMethodAvailableChoice.nms
    sigma: Optional[confloat(gt=0, le=1)] = 0.3
    line_thickness: Optional[conint(ge=1, le=5)] = 1
    filter_classes: Optional[List[str]] = ObjectDetectionFilterClassesList[0]

    class_id: Optional[conint(ge=0)] = 0
    classes_colors: Optional[List[Color]]
    open_tag: Optional[List[str]]
    close_tag: Optional[List[str]]
    classes: Optional[List[str]]
    colors: Optional[List[Color]]

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
        elif _type == BlockFunctionTypeChoice.PlotBboxes:
            _keys += ["classes", "colors", "line_thickness"]
        elif _type == BlockFunctionTypeChoice.FilterClasses:
            _keys += ["filter_classes"]
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
        elif value == BlockFunctionTypeChoice.PutTag:
            cls.__fields__["alpha"].required = True
        elif value == BlockFunctionTypeChoice.PostprocessBoxes:
            cls.__fields__["score_threshold"].required = True
            cls.__fields__["iou_threshold"].required = True
            cls.__fields__["method"].required = True
            if (
                cls.__fields__["method"]
                == PostprocessBoxesMethodAvailableChoice.soft_nms
            ):
                cls.__fields__["sigma"].required = True
        elif value == BlockFunctionTypeChoice.PlotBboxes:
            cls.__fields__["line_thickness"].required = True
        return value

    @validator(
        "shape",
        "classes_colors",
        "open_tag",
        "close_tag",
        "classes",
        "colors",
        "filter_classes",
        pre=True,
    )
    def _validate_empty_list(cls, value):
        if not value:
            value = None
        return value
