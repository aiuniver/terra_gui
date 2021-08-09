from enum import Enum
from typing import Optional, List, Union
from pydantic.types import DirectoryPath, PositiveInt, PositiveFloat, FilePath

from pydantic import validator

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice, LayerAudioParameterChoice


class AudioModeChoice(str, Enum):
    completely = 'Целиком'
    length_and_step = 'По длине и шагу'


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    cols_names: Optional[List[str]]
    audio_mode: AudioModeChoice = AudioModeChoice.completely
    sample_rate: PositiveInt
    max_seconds: PositiveFloat
    length: PositiveFloat
    step: PositiveFloat
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    parameter: LayerAudioParameterChoice = LayerAudioParameterChoice.audio_signal

    @validator("audio_mode", allow_reuse=True)
    def _validate_prepare_method(
            cls, value: AudioModeChoice
    ) -> AudioModeChoice:
        if value == AudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        else:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
