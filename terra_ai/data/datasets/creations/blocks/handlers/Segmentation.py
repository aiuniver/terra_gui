from typing import List, Optional
from pydantic import validator, PositiveInt
from pydantic.color import Color

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import ConstrainedIntValueGe0


class OptionsData(BaseMixinData):
    mask_range: ConstrainedIntValueGe0
    classes_names: List[str]
    classes_colors: List[Color]

    # Внутренние параметры
    width: Optional[PositiveInt]
    height: Optional[PositiveInt]

    @validator("width", "height", pre=True)
    def _validate_empty_number(cls, value: PositiveInt) -> PositiveInt:
        if not value:
            value = None
        return value
