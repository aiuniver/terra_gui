from ...extra import SourcesPathsData
from .....extra import LayerYoloChoice
from pydantic.types import PositiveInt
from typing import Optional


class ParametersData(SourcesPathsData):
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    put: Optional[PositiveInt]
    cols_names: Optional[str]
