from typing import List
from typing import Optional
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
