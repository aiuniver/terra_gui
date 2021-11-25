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
from terra_ai.data.datasets.extra import LayerYoloChoice, LayerODDatasetTypeChoice, LayerObjectDetectionModelChoice
from terra_ai.data.datasets.creations.layers.extra import MinMaxScalerData, LayerImageFrameModeChoice

class ParametersBaseData(BaseMixinData):
    pass


class ParametersImageData(ParametersBaseData, MinMaxScalerData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerImageChoice
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch


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
    open_tags: Optional[str]
    close_tags: Optional[str]

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
    frame_mode: LayerVideoFrameModeChoice = LayerVideoFrameModeChoice.fit
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


class ParametersScalerData(ParametersBaseData, MinMaxScalerData):
    scaler: LayerScalerDefaultChoice
    length: int = 0
    depth: int = 0
    step: int = 1

    def __init__(self, **data):
        try:
            data.pop("length")
            data.pop("depth")
            data.pop("step")
        except KeyError:
            pass
        super().__init__(**data)


class ParametersClassificationData(ParametersBaseData):
    one_hot_encoding: bool = True
    type_processing: Optional[
        LayerTypeProcessingClassificationChoice
    ] = LayerTypeProcessingClassificationChoice.categorical
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


class ParametersTextSegmentationData(ParametersBaseData):
    open_tags: Optional[str]
    close_tags: Optional[str]

    sources_paths: Optional[list]
    filters: Optional[str]
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]


class ParametersObjectDetectionData(ParametersBaseData):
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo_terra
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch


class ParametersRegressionData(ParametersBaseData, MinMaxScalerData):
    scaler: LayerScalerRegressionChoice


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


class ParametersTimeseriesData(ParametersBaseData, MinMaxScalerData):
    length: PositiveInt
    step: PositiveInt
    trend: bool
    trend_limit: Optional[str]
    depth: Optional[PositiveInt]
    scaler: Optional[LayerScalerTimeseriesChoice]

    def __init__(self, **data):
        try:
            if data.get("trend"):
                data.pop("depth")
                data.pop("scaler")
            else:
                data.pop("trend_limit")
        except KeyError:
            pass
        super().__init__(**data)

    @validator("trend")
    def _validate_trend(cls, value: bool) -> bool:
        if value:
            cls.__fields__["trend_limit"].required = True
        else:
            cls.__fields__["depth"].required = True
            cls.__fields__["scaler"].required = True
        return value
