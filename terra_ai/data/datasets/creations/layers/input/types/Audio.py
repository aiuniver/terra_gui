from enum import Enum
from typing import Optional
from pydantic.types import PositiveInt, PositiveFloat
from pydantic import validator

from terra_ai.data.datasets.creations.layers.extra import FileInfo
from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice


class AudioModeChoice(str, Enum):
    completely = 'Целиком'
    length_and_step = 'По длине и шагу'


class ParametersData(BaseMixinData):
    file_info: FileInfo
    audio_mode: AudioModeChoice = AudioModeChoice.completely
    sample_rate: PositiveInt
    max_seconds: PositiveFloat
    length: PositiveFloat
    step: PositiveFloat
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False

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
