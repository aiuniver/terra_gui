"""
## Тип слоя `Rescaling`
"""

from typing import Optional
from pydantic.types import PositiveFloat

from ...mixins import BaseMixinData
from ...types import ConstrainedFloatValueGe0Le1
from .extra import InitializerChoice, RegularizerChoice, ConstraintChoice


class ParametersMainData(BaseMixinData):
    scale: float = 1.0
    offset: float = 0.0


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers.experimental.preprocessing'
    module_type: str = 'keras'

