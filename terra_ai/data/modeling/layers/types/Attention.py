"""
## Тип слоя `Attention`
"""

from ...mixins import BaseMixinData
from .extra import ActivationChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = [2, 3]
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
