from typing import Optional
from pydantic.types import PositiveInt
from terra_ai.data.datasets.creations.layers.extra import SourcesPathsData, ColumnProcessingData


class ParametersData(SourcesPathsData, ColumnProcessingData):
    put: Optional[PositiveInt]
