"""
## Тип слоя `Average`
"""

from ....mixins import BaseMixinData
from ..extra import ActivationChoice


class ParametersMainData(BaseMixinData):
    activation: ActivationChoice = ActivationChoice.relu


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = '2+'
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
