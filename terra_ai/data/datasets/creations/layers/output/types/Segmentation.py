from typing import List

from pydantic.types import PositiveInt
from pydantic.color import Color

from ...extra import SourcesPathsData, ColumnProcessingData


class ParametersData(SourcesPathsData, ColumnProcessingData):
    mask_range: PositiveInt
    classes_names: List[str]
    classes_colors: List[Color]
