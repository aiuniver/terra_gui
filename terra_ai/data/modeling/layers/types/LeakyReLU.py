"""
## Тип слоя `LeakyReLU`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0


class ParametersMainData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0 = 0.3


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
