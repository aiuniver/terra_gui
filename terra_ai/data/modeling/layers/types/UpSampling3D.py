"""
## Тип слоя `UpSampling3D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 5
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
