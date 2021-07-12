"""
## Тип слоя `GlobalMaxPool3D`
"""

from typing import Optional

from pydantic import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 5
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 5
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras