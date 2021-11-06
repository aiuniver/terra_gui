from typing import Optional, List, Union, Dict
from pydantic import validator
from pydantic.types import DirectoryPath, PositiveInt, PositiveFloat

from .image_augmentation import AugmentationData
from ...extra import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerPrepareMethodChoice,
    LayerTextModeChoice,
    LayerScalerAudioChoice,
    LayerAudioParameterChoice,
    LayerAudioModeChoice,
    LayerAudioResampleChoice,
    LayerAudioFillModeChoice, LayerImageFrameModeChoice,
)
from ....mixins import BaseMixinData
from ....types import confilepath


class SourcesPathsData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, confilepath(ext="csv")]]


class MinMaxScalerData(BaseMixinData):
    min_scaler: int = 0
    max_scaler: int = 1


class ColumnProcessingData(BaseMixinData):
    cols_names: Dict[int, List[int]] = {}


class ParametersImageData(MinMaxScalerData, SourcesPathsData, ColumnProcessingData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice
    scaler: LayerScalerImageChoice
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch

    put: Optional[PositiveInt]
    augmentation: Optional[AugmentationData]
    deploy: Optional[bool] = False
    object_detection: Optional[bool] = False


class ParametersTextData(SourcesPathsData, ColumnProcessingData):
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    text_mode: LayerTextModeChoice
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    pymorphy: bool = False
    prepare_method: LayerPrepareMethodChoice
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]

    put: Optional[PositiveInt]
    deploy: Optional[bool] = False
    open_tags: Optional[str]
    close_tags: Optional[str]

    def __init__(self, **data):
        data.update({"cols_names": {}})
        super().__init__(**data)

    @validator("text_mode")
    def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
            cls.__fields__["length"].required = False
            cls.__fields__["step"].required = False
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["max_words"].required = False
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("prepare_method")
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value in [
            LayerPrepareMethodChoice.embedding,
            LayerPrepareMethodChoice.bag_of_words,
        ]:
            cls.__fields__["max_words_count"].required = True
            cls.__fields__["word_to_vec_size"].required = False
        elif value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["max_words_count"].required = False
            cls.__fields__["word_to_vec_size"].required = True
        return value


class ParametersAudioData(MinMaxScalerData, SourcesPathsData, ColumnProcessingData):
    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    parameter: LayerAudioParameterChoice
    fill_mode: LayerAudioFillModeChoice
    resample: LayerAudioResampleChoice
    scaler: LayerScalerAudioChoice
    put: Optional[PositiveInt]

    deploy: Optional[bool] = False

    @validator("audio_mode")
    def _validate_audio_mode(cls, value: LayerAudioModeChoice) -> LayerAudioModeChoice:
        if value == LayerAudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
            cls.__fields__["length"].required = False
            cls.__fields__["step"].required = False
        elif value == LayerAudioModeChoice.length_and_step:
            cls.__fields__["max_seconds"].required = False
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


class ParametersDataframeData(SourcesPathsData, ColumnProcessingData):
    pass
