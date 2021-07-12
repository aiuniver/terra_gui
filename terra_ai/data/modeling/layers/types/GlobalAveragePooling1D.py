"""
## Тип слоя `GlobalAveragePooling1D`
"""
from pydantic import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 3
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
