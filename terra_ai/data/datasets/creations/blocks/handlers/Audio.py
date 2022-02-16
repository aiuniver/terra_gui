from typing import Optional
from pydantic import validator, PositiveInt, PositiveFloat

from terra_ai.data.datasets.extra import (
    LayerAudioModeChoice,
    LayerAudioParameterChoice,
    LayerAudioFillModeChoice,
    LayerAudioResampleChoice,
    LayerScalerAudioChoice,
)
from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
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
