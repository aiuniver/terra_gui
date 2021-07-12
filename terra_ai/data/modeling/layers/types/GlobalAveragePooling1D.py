"""
## Тип слоя `GlobalAveragePooling1D`
"""

from ...mixins import BaseMixinData
from .extra import DataFormatChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 3
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
