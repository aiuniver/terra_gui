from typing import Optional
from pydantic.types import PositiveInt

from ...extra import SourcesPathsData, ColumnProcessingData


class ParametersData(SourcesPathsData, ColumnProcessingData):
    put: Optional[PositiveInt]
