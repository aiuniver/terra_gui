from enum import Enum
from typing import Optional, List
from pydantic.types import PositiveInt, PositiveFloat

from pydantic import validator

from ...extra import ParametersBaseData
from .....extra import LayerScalerChoice, LayerAudioParameterChoice


class AudioModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    audio_mode: AudioModeChoice = AudioModeChoice.completely
    sample_rate: PositiveInt
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    parameter: LayerAudioParameterChoice = LayerAudioParameterChoice.audio_signal
    deploy: Optional[bool] = False

    @validator("audio_mode", allow_reuse=True)
    def _validate_audio_mode(cls, value: AudioModeChoice) -> AudioModeChoice:
        if value == AudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        else:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
