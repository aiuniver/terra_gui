"""
## Тип слоя `MaxPool3D`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import (
    PaddingChoice,
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
            "value": 5,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    pool_size: Tuple[PositiveInt, PositiveInt, PositiveInt] = (2, 2, 2)
    strides: Optional[Tuple[PositiveInt, PositiveInt, PositiveInt]]
    padding: PaddingChoice = PaddingChoice.same


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
