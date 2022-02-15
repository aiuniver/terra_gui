from terra_ai.data.mixins import BaseMixinData
from pydantic.types import PositiveInt
from typing import Optional


class ParametersData(BaseMixinData):
    # Внутренние параметры
    put: Optional[PositiveInt]
