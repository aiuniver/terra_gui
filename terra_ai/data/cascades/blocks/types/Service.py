from typing import Optional
from pydantic import PositiveInt, validator
from pydantic.types import confloat

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import (
    BlockServiceGroupChoice,
    BlockServiceTypeChoice,
    BlockServiceBiTBasedTrackerMetricChoice,
    BlockServiceYoloV5VersionChoice,
    BlockServiceGoogleTTSLanguageChoice,
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
    max_dist: Optional[confloat(gt=0, lt=1)] = 0.2
    min_confidence: Optional[confloat(gt=0, lt=1)] = 0.3
    nms_max_overlap: Optional[confloat(gt=0, le=1)] = 1
    max_iou_distance: Optional[confloat(gt=0, lt=1)] = 0.7
    deep_max_age: Optional[PositiveInt] = 70
    n_init: Optional[PositiveInt] = 3
    nn_budget: Optional[PositiveInt] = 100
    api_key: Optional[str]
    secret_key: Optional[str]
    max_alternatives: Optional[PositiveInt] = 3
    do_not_perform_vad: Optional[bool] = True
    profanity_filter: Optional[bool] = True
    expiration_time: Optional[PositiveInt] = 60000
    endpoint: Optional[str]
    language: Optional[
        BlockServiceGoogleTTSLanguageChoice
    ] = BlockServiceGoogleTTSLanguageChoice.ru

    model_path: Optional[str]

    def __init__(self, **data):
        _type = data.get("type")
        _keys = ["group", "type"]
        if _type == BlockServiceTypeChoice.Sort:
            _keys += ["max_age", "min_hits"]
        elif _type == BlockServiceTypeChoice.BiTBasedTracker:
            _keys += ["max_age", "distance_threshold", "metric"]
        elif _type == BlockServiceTypeChoice.DeepSort:
            _keys += [
                "max_dist",
                "min_confidence",
                "nms_max_overlap",
                "max_iou_distance",
                "deep_max_age",
                "n_init",
                "nn_budget",
            ]
        elif _type == BlockServiceTypeChoice.GoogleSTT:
            _keys += []
        elif _type == BlockServiceTypeChoice.GoogleTTS:
            _keys += ["language"]
        elif _type == BlockServiceTypeChoice.Wav2Vec:
            _keys += []
        elif _type == BlockServiceTypeChoice.Google:
            _keys += []
        elif _type == BlockServiceTypeChoice.TinkoffAPI:
            _keys += [
                "api_key",
                "secret_key",
                "max_alternatives",
                "do_not_perform_vad",
                "profanity_filter",
                "expiration_time",
                "endpoint",
            ]
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
        if value == BlockServiceTypeChoice.DeepSort:
            cls.__fields__["max_dist"].required = True
            cls.__fields__["min_confidence"].required = True
            cls.__fields__["nms_max_overlap"].required = True
            cls.__fields__["max_iou_distance"].required = True
            cls.__fields__["deep_max_age"].required = True
            cls.__fields__["n_init"].required = True
            cls.__fields__["nn_budget"].required = True
        if value == BlockServiceTypeChoice.GoogleSTT:
            pass
        if value == BlockServiceTypeChoice.GoogleTTS:
            cls.__fields__["language"].required = True
        if value == BlockServiceTypeChoice.Wav2Vec:
            pass
        if value == BlockServiceTypeChoice.Google:
            pass
        if value == BlockServiceTypeChoice.TinkoffAPI:
            cls.__fields__["api_key"].required = True
            cls.__fields__["secret_key"].required = True
            cls.__fields__["max_alternatives"].required = True
            cls.__fields__["expiration_time"].required = True
            cls.__fields__["endpoint"].required = True
        if value == BlockServiceTypeChoice.YoloV5:
            cls.__fields__["version"].required = True

        return value
