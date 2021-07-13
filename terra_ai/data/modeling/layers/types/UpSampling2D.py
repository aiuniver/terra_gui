"""
## Тип слоя `UpSampling2D`
"""

from typing import Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice, InterpolationChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 4
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    size: Tuple[PositiveInt, PositiveInt] = (2, 2)


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
    interpolation: InterpolationChoice = InterpolationChoice.nearest
