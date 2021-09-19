from typing import List

from terra_ai.data.types import confilepath
from terra_ai.data.datasets.creations.layers.extra import ColumnProcessingData


class ParametersData(ColumnProcessingData):
    sources_paths: List[confilepath(ext="csv")]
