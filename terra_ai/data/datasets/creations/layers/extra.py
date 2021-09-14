from typing import Optional, List, Union
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
)
from ....mixins import BaseMixinData
from ....types import confilepath


class SourcesPathsData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, confilepath(ext="csv")]]


class MinMaxScalerData(BaseMixinData):
    min_scaler: int = 0
    max_scaler: int = 1


class ParametersImageData(MinMaxScalerData, SourcesPathsData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice
    scaler: LayerScalerImageChoice

    put: Optional[str]
    cols_names: Optional[List[str]]
    augmentation: Optional[AugmentationData]
    deploy: Optional[bool] = False
    object_detection: Optional[bool] = False


class ParametersTextData(SourcesPathsData):
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    text_mode: LayerTextModeChoice
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    pymorphy: bool = False
    prepare_method: LayerPrepareMethodChoice
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]

    put: Optional[str]
    deploy: Optional[bool] = False
    cols_names: Optional[List[str]]
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


class ParametersAudioData(MinMaxScalerData, SourcesPathsData):
    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    parameter: LayerAudioParameterChoice
    scaler: LayerScalerAudioChoice

    cols_names: Optional[List[str]]
    deploy: Optional[bool] = False

    @validator("audio_mode")
    def _validate_audio_mode(cls, value: LayerAudioModeChoice) -> LayerAudioModeChoice:
        if value == LayerAudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        elif value == LayerAudioModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
