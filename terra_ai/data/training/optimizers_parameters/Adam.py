"""
## Оптимайзер `Adam`
"""

from pydantic.types import PositiveFloat

from ...mixins import BaseMixinData


class ParametersExtraData(BaseMixinData):
    beta_1: PositiveFloat = 0.9
    beta_2: PositiveFloat = 0.999
    epsilon: PositiveFloat = 1e-07
    amsgrad: bool = False
