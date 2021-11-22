from typing import Optional
from pydantic import PositiveInt, validator
from pydantic.types import confloat

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import (
    BlockServiceGroupChoice,
    BlockServiceTypeChoice,
    BlockServiceBiTBasedTrackerMetricChoice,
    BlockServiceYoloV5VersionChoice,
)


class ParametersMainData(BaseMixinData):
    group: BlockServiceGroupChoice
    type: BlockServiceTypeChoice
    max_age: Optional[PositiveInt] = 4
    min_hits: Optional[PositiveInt] = 4
    distance_threshold: Optional[confloat(gt=0, lt=1)] = 0.4
    metric: Optional[
        BlockServiceBiTBasedTrackerMetricChoice
    ] = BlockServiceBiTBasedTrackerMetricChoice.euclidean
    version: Optional[
        BlockServiceYoloV5VersionChoice
    ] = BlockServiceYoloV5VersionChoice.Small
    render_img: Optional[bool] = True

    def __init__(self, **data):
        _type = data.get("type")
        _keys = ["group", "type"]
        if _type == BlockServiceTypeChoice.Sort:
            _keys += ["max_age", "min_hits"]
        elif _type == BlockServiceTypeChoice.BiTBasedTracker:
            _keys += ["max_age", "distance_threshold", "metric"]
        elif _type == BlockServiceTypeChoice.YoloV5:
            _keys += ["version", "render_img"]
        data = dict(filter(lambda item: item[0] in _keys, data.items()))
        super().__init__(**data)

    @validator("type")
    def _validate_type(cls, value: BlockServiceTypeChoice) -> BlockServiceTypeChoice:
        for name, item in cls.__fields__.items():
            if name in ["group", "type"]:
                continue
            cls.__fields__[name].required = False
        if value == BlockServiceTypeChoice.Sort:
            cls.__fields__["max_age"].required = True
            cls.__fields__["min_hits"].required = True
        if value == BlockServiceTypeChoice.BiTBasedTracker:
            cls.__fields__["max_age"].required = True
            cls.__fields__["distance_threshold"].required = True
            cls.__fields__["metric"].required = True
        if value == BlockServiceTypeChoice.YoloV5:
            cls.__fields__["version"].required = True

        return value
