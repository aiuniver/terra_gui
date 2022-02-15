from terra_ai.data.datasets.creations.layers.extra import MinMaxScalerData
from terra_ai.data.datasets.extra import LayerAudioModeChoice, LayerAudioParameterChoice, LayerAudioFillModeChoice, \
    LayerAudioResampleChoice, LayerScalerAudioChoice
from terra_ai.data.mixins import BaseMixinData
from pydantic.types import PositiveInt, PositiveFloat
from pydantic import validator
from typing import Optional


class ParametersAudioData(BaseMixinData, MinMaxScalerData):
    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    parameter: LayerAudioParameterChoice
    fill_mode: LayerAudioFillModeChoice
    resample: LayerAudioResampleChoice
    scaler: LayerScalerAudioChoice

    # Внутренние параметры
    deploy: Optional[bool] = False
    put: Optional[PositiveInt]

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