"""
## Тип слоя `Dropout`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe2


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    rate: ConstrainedFloatValueGe0Le1 = 0.1


class ParametersExtraData(BaseMixinData):
    noise_shape: Optional[PositiveInt]
    seed: Optional[PositiveInt]
