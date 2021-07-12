"""
## Тип слоя `AveragePooling1D`
"""

from typing import Optional

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import PaddingChoice, DataFormatChoice


class ParametersMainData(BaseMixinData):
    pool_size: PositiveInt = 2
    strides: Optional[PositiveInt]
    padding: PaddingChoice = PaddingChoice.valid


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 3
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
