"""
## Тип слоя `ReLU`
"""

from typing import Optional

from pydantic import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0, ConstrainedIntValueGe2


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    max_value: Optional[ConstrainedFloatValueGe0]
    negative_slope: ConstrainedFloatValueGe0 = 0
    threshold: ConstrainedFloatValueGe0 = 0
