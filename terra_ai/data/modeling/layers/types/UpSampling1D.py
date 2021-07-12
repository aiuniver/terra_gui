"""
## Тип слоя `UpSampling1D`
"""

from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    size: PositiveInt = 2


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: PositiveInt = 3
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras