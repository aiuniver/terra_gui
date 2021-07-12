"""
## Тип слоя `ThresholdedReLU`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    theta: float = 1.0


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers'
    module_type: str = 'keras'
