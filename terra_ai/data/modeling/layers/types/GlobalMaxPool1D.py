"""
## Тип слоя `GlobalMaxPool1D`
"""

from ....mixins import BaseMixinData
from ..extra import DataFormatChoice


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: PositiveInt = 3
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
