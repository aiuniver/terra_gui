"""
## Тип слоя `Resizing`
"""
from typing import Optional

from pydantic import PositiveInt

from ..extra import ModuleChoice, ModuleTypeChoice
from ....mixins import BaseMixinData
from ..extra import ResizingInterpolationChoice
from ....types import ConstrainedIntValueGe2


# class LayerConfig(BaseMixinData):
#     num_uplinks: PositiveInt = 1
#     input_dimension: ConstrainedIntValueGe2 = 2
#     module: ModuleChoice = ModuleChoice.tensorflow_keras_layers_preprocessing
#     module_type: ModuleTypeChoice = ModuleTypeChoice.keras


class ParametersMainData(BaseMixinData):
    height: PositiveInt = 224
    width: PositiveInt = 224


class ParametersExtraData(BaseMixinData):
    interpolation: ResizingInterpolationChoice = ResizingInterpolationChoice.bilinear
