"""
## Тип слоя `ELU`
"""

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0Le1


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    alpha: ConstrainedFloatValueGe0Le1 = 1


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
