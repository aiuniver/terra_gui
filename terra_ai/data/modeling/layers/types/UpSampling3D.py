"""
## Тип слоя `UpSampling3D`
"""

from typing import Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 5
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
