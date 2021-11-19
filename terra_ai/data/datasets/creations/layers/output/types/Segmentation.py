from typing import List, Optional
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.color import Color

from ...extra import SourcesPathsData, ColumnProcessingData


class ParametersData(SourcesPathsData, ColumnProcessingData):
    mask_range: PositiveInt
    classes_names: List[str]
    classes_colors: List[Color]
    width: Optional[PositiveInt]
    height: Optional[PositiveInt]
    put: Optional[PositiveInt]

    @validator("width", "height", "put", pre=True)
    def _validate_empty_number(cls, value: PositiveInt) -> PositiveInt:
        if not value:
            value = None
        return value
