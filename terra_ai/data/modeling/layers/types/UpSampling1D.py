"""
## Тип слоя `UpSampling1D`
"""

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 3
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    size: PositiveInt = 2


class ParametersExtraData(BaseMixinData):
    pass
