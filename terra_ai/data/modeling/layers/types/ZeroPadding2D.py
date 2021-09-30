"""
## Тип слоя `ZeroPadding2D`
"""

from typing import Tuple

from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe0
from ..extra import (
    DataFormatChoice,
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
)


LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 4,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    padding: Tuple[
        Tuple[ConstrainedIntValueGe0, ConstrainedIntValueGe0],
        Tuple[ConstrainedIntValueGe0, ConstrainedIntValueGe0],
    ] = ((1, 0), (1, 0))


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
