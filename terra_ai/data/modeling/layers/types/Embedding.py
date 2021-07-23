"""
## Тип слоя `Embedding`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ..extra import (
    InitializerChoice,
    RegularizerChoice,
    ConstraintChoice,
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
            "value": 2,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    input_dim: PositiveInt = 20000
    output_dim: PositiveInt = 64


class ParametersExtraData(BaseMixinData):
    embeddings_initializer: InitializerChoice = InitializerChoice.uniform
    embeddings_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    embeddings_constraint: Optional[ConstraintChoice]
    mask_zero: bool = False
    input_length: Optional[PositiveInt]
