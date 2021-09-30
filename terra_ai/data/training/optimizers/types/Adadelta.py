"""
## Оптимайзер `Adadelta`
"""

from pydantic.types import PositiveFloat

from ....mixins import BaseMixinData


class ParametersExtraData(BaseMixinData):
    rho: PositiveFloat = 0.95
    epsilon: PositiveFloat = 1e-07
