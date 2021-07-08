"""
## Тип слоя `Embedding`
"""

from typing import Optional
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from .extra import InitializerChoice, RegularizerChoice, ConstraintChoice


class ParametersMainData(BaseMixinData):
    input_dim: PositiveInt
    output_dim: PositiveInt


class ParametersExtraData(BaseMixinData):
    embeddings_initializer: InitializerChoice = InitializerChoice.uniform
    embeddings_regularizer: Optional[RegularizerChoice]
    activity_regularizer: Optional[RegularizerChoice]
    embeddings_constraint: Optional[ConstraintChoice]
    mask_zero: bool = False
    input_length: Optional[PositiveInt]
