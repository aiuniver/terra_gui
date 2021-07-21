"""
## Тип слоя `PReLU`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import InitializerChoice, RegularizerChoice, ConstraintChoice, LayerConfigData, \
    LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    alpha_initializer: InitializerChoice = InitializerChoice.zeros
    alpha_regularizer: Optional[RegularizerChoice]
    alpha_constraint: Optional[ConstraintChoice]
    shared_axes: Optional[Tuple[PositiveInt, ...]] = None
