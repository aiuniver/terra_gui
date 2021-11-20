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
    metric: Optional[BlockServiceBiTBasedTrackerMetricChoice] = BlockServiceBiTBasedTrackerMetricChoice.euclidean
    version: Optional[BlockServiceYoloV5VersionChoice] = BlockServiceYoloV5VersionChoice.Small
    render_img: Optional[bool] = True

    @validator("type")
    def _validate_type(cls, value: BlockServiceTypeChoice) -> BlockServiceTypeChoice:
        if value == BlockServiceTypeChoice.Sort:
            cls.__fields__["max_age"].required = True
            cls.__fields__["min_hits"].required = True
        if value == BlockServiceTypeChoice.BiTBasedTracker:
            cls.__fields__["max_age"].required = True
            cls.__fields__["distance_threshold"].required = True
            cls.__fields__["metric"].required = True
        if value == BlockServiceTypeChoice.YoloV5:
            cls.__fields__["version"].required = True
            cls.__fields__["render_img"].required = True
     
        return value
