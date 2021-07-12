"""
## Тип слоя `Concatenate`
"""

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    axis: int = -1


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = '2+'
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'