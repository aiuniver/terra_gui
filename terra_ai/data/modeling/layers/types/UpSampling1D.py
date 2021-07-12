"""
## Тип слоя `UpSampling1D`
"""

from pydantic.types import PositiveInt

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    size: PositiveInt = 2


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = 3
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
