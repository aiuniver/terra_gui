from typing import Optional
from pydantic.types import DirectoryPath, PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerScalerChoice


class ParametersData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: PositiveInt
    step: PositiveInt
    scaler: LayerScalerChoice = LayerScalerChoice.no_scaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False
