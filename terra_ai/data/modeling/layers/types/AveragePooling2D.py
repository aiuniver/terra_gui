"""
## Тип слоя `AveragePooling2D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import PaddingChoice, DataFormatChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 4
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    pool_size: Tuple[PositiveInt, PositiveInt] = (2, 2)
    strides: Optional[Tuple[PositiveInt, PositiveInt]]
    padding: PaddingChoice = PaddingChoice.same


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
