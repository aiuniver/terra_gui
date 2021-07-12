"""
## Тип слоя `Dropout`
"""

from typing import Optional
from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    rate: ConstrainedFloatValueGe0Le1 = 0.1


class ParametersExtraData(BaseMixinData):
    noise_shape: Optional[PositiveInt]
    seed: Optional[PositiveInt]


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras