"""
## Тип слоя `MaxPool2D`
"""

from typing import Optional, Tuple

from pydantic import validator
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import PaddingChoice, DataFormatChoice


class ParametersMainData(BaseMixinData):
    pool_size: Tuple[PositiveInt, PositiveInt] = (2, 2)
    strides: Optional[PositiveInt]
    padding: PaddingChoice = PaddingChoice.same

    # @validator('pool_size')
    # def pool_size_validate(cls, pool_size):
    #     if pool_size <= 0:
    #         raise ValueError("must be positive integer")
    #     return pool_size


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 4
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
