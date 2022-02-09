from pydantic.types import PositiveInt
from typing import Optional

from .....extra import (
    LayerYoloChoice,
    LayerODDatasetTypeChoice,
    LayerObjectDetectionModelChoice,
)
from ...extra import SourcesPathsData, LayerImageFrameModeChoice


class ParametersData(SourcesPathsData):
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    # put: Optional[PositiveInt]
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo_terra
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    # cols_names: Optional[str]

    # def __init__(self, **data):
    #     data.update({"cols_names": None})
    #     super().__init__(**data)
