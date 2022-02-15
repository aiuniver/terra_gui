from terra_ai.data.mixins import BaseMixinData
from typing import Optional
from pydantic.types import PositiveInt


class ParametersGeneratorData(BaseMixinData):
    # Внутренние параметры
    shape: tuple
    put: Optional[PositiveInt]
    deploy: Optional[bool] = False
