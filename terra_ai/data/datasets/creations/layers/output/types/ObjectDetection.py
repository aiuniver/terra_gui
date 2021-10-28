from ...extra import SourcesPathsData
from .....extra import LayerYoloChoice, LayerObjectDetectionModelChoice
from pydantic.types import PositiveInt
from typing import Optional
from enum import Enum


class LayerODDatasetTypeChoice(str, Enum):
    Kitti = "Kitti"
    Voc = "Voc"
    Coco = "Coco"
    Udacity = "Udacity"
    Yolo = "Yolo"


class ParametersData(SourcesPathsData):
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    put: Optional[PositiveInt]
    cols_names: Optional[str]
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo

    def __init__(self, **data):
        data.update({"cols_names": None})
        super().__init__(**data)
