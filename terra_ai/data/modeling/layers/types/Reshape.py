"""
## Тип слоя `Reshape`
"""

from typing import Tuple

from pydantic import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    target_shape: Tuple[int, ...] = ()


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
