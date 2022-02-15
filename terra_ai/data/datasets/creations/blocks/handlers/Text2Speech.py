from typing import Optional
from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData


class OptionsData(BaseMixinData):

    # Внутренние параметры
    put: Optional[PositiveInt]
