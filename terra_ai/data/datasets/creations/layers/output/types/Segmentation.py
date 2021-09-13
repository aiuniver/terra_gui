from typing import Optional, List

from pydantic.types import PositiveInt
from pydantic.color import Color

from ...extra import SourcesPathsData


class ParametersData(SourcesPathsData):
    mask_range: PositiveInt
    classes_names: List[str]
    classes_colors: List[Color]
    width: Optional[PositiveInt]
    height: Optional[PositiveInt]

    cols_names: Optional[List[str]]
