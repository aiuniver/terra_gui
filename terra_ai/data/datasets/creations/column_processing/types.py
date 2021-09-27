from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt, PositiveFloat
from pydantic.color import Color

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerScalerVideoChoice,
    LayerScalerRegressionChoice,
    LayerScalerTimeseriesChoice,
    LayerScalerDefaultChoice,
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
    LayerAudioModeChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
    LayerTypeProcessingClassificationChoice,
)
from terra_ai.data.datasets.creations.layers.extra import MinMaxScalerData


class ParametersBaseData(BaseMixinData):
    pass


class ParametersImageData(ParametersBaseData, MinMaxScalerData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerImageChoice


class ParametersTextData(ParametersBaseData):
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    text_mode: LayerTextModeChoice = LayerTextModeChoice.completely
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    pymorphy: bool = False
    prepare_method: LayerPrepareMethodChoice
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]

    @validator("text_mode")
    def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("prepare_method")
    def _validate_prepare_method(cls, value: bool) -> bool:
        if value in [
            LayerPrepareMethodChoice.embedding,
            LayerPrepareMethodChoice.bag_of_words,
        ]:
            cls.__fields__["max_words_count"].required = True
        elif value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class ParametersAudioData(ParametersBaseData, MinMaxScalerData):
    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice = LayerAudioModeChoice.completely
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    fill_mode: LayerAudioFillModeChoice = LayerAudioFillModeChoice.last_millisecond
    parameter: LayerAudioParameterChoice
    resample: LayerAudioResampleChoice = LayerAudioResampleChoice.kaiser_best
    scaler: LayerScalerAudioChoice

    @validator("audio_mode")
    def _validate_audio_mode(cls, value: LayerAudioModeChoice) -> LayerAudioModeChoice:
        if value == LayerAudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        elif value == LayerAudioModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


class ParametersVideoData(ParametersBaseData, MinMaxScalerData):
    width: PositiveInt
    height: PositiveInt
    fill_mode: LayerVideoFillModeChoice = LayerVideoFillModeChoice.last_frames
    frame_mode: LayerVideoFrameModeChoice = LayerVideoFrameModeChoice.keep_proportions
    video_mode: LayerVideoModeChoice = LayerVideoModeChoice.completely
    max_frames: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    scaler: LayerScalerVideoChoice

    @validator("video_mode")
    def _validate_video_mode(cls, value: LayerVideoModeChoice) -> LayerVideoModeChoice:
        if value == LayerVideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        elif value == LayerVideoModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


class ParametersSegmentationData(ParametersBaseData):
    mask_range: PositiveInt
    classes_names: List[str]
    classes_colors: List[Color]
    height: Optional[PositiveInt]
    width: Optional[PositiveInt]

    @validator("width", "height", pre=True)
    def _validate_size(cls, value: PositiveInt) -> PositiveInt:
        if not value:
            value = None
        return value


class ParametersClassificationData(ParametersBaseData):
    one_hot_encoding: bool = True
    type_processing: LayerTypeProcessingClassificationChoice
    ranges: Optional[str]
    length: int = 0
    depth: int = 0
    step: int = 1

    @validator("type_processing")
    def _validate_type_processing(
        cls, value: LayerTypeProcessingClassificationChoice
    ) -> LayerTypeProcessingClassificationChoice:
        if value == LayerTypeProcessingClassificationChoice.ranges:
            cls.__fields__["ranges"].required = True
        return value


class ParametersRegressionData(ParametersBaseData, MinMaxScalerData):
    scaler: LayerScalerRegressionChoice


class ParametersTimeseriesData(ParametersBaseData, MinMaxScalerData):
    length: PositiveInt
    step: PositiveInt
    trend: bool
    trend_limit: Optional[str]
    depth: Optional[PositiveInt]
    scaler: Optional[LayerScalerTimeseriesChoice]

    @validator("trend")
    def _validate_trend(cls, value: bool) -> bool:
        if value:
            cls.__fields__["trend_limit"].required = True
        else:
            cls.__fields__["depth"].required = True
            cls.__fields__["scaler"].required = True
        return value


class ParametersScalerData(ParametersBaseData, MinMaxScalerData):
    scaler: LayerScalerDefaultChoice
    length: int = 0
    depth: int = 0
    step: int = 1
