"""
## Тип слоя `RepeatVector`
"""

from pydantic.types import PositiveInt

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    n: PositiveInt


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 2
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
