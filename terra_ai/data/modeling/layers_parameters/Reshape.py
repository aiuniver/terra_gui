"""
## Тип слоя `Reshape`
"""

from typing import Tuple

from ...mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    target_shape: Tuple[int, ...] = ()
