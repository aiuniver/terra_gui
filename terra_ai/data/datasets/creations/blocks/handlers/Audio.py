from typing import Optional
from pydantic import validator, PositiveInt, PositiveFloat

from terra_ai.data.datasets.extra import (
    LayerAudioModeChoice,
    LayerAudioParameterChoice,
    LayerAudioFillModeChoice,
    LayerAudioResampleChoice,
)
from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice
    max_seconds: Optional[PositiveFloat] = 1
    length: Optional[PositiveFloat] = 1
    step: Optional[PositiveFloat] = 0.5
    fill_mode: LayerAudioFillModeChoice
    parameter: LayerAudioParameterChoice
    resample: LayerAudioResampleChoice

    # Внутренние параметры
    deploy: Optional[bool] = False

    @validator("audio_mode")
    def _validate_audio_mode(cls, value):
        if value == LayerAudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        elif value == LayerAudioModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
