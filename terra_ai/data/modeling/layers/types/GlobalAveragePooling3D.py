"""
## Тип слоя `GlobalAveragePooling3D`
"""

from typing import Optional

from pydantic import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 5
    module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
