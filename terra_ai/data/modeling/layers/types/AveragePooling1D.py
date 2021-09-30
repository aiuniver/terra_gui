"""
## Тип слоя `AveragePooling1D`
"""

from typing import Optional

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import PaddingChoice, DataFormatChoice, LayerConfigData, LayerValidationMethodChoice, ModuleChoice, \
    ModuleTypeChoice


LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 3,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    pool_size: PositiveInt = 2
    strides: Optional[PositiveInt]
    padding: PaddingChoice = PaddingChoice.valid


class ParametersExtraData(BaseMixinData):
    data_format: DataFormatChoice = DataFormatChoice.channels_last
