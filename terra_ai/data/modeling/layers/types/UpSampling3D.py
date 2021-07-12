"""
## Тип слоя `UpSampling3D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 5
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
