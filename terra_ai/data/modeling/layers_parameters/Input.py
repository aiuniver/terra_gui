"""
## Тип слоя `Input`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    shape: Optional[Tuple[PositiveInt, ...]]
    batch_size: Optional[PositiveInt]
    name: Optional[str]
    dtype: Optional[str]
    sparse: Optional[bool]
    tensor: Optional[str]
    ragged: Optional[bool]
    type_spec: Optional[str]


class ParametersExtraData(BaseMixinData):
    pass
