from terra_ai.data.datasets.extra import LayerYoloChoice, LayerODDatasetTypeChoice, LayerObjectDetectionModelChoice, \
    LayerImageFrameModeChoice
from terra_ai.data.mixins import BaseMixinData
from pydantic.types import PositiveInt
from typing import Optional


class ParametersData(BaseMixinData):
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo_terra
    # Внутренние параметры
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    put: Optional[PositiveInt]
    # cols_names: Optional[str]

    # def __init__(self, **data):
    #     data.update({"cols_names": None})
    #     super().__init__(**data)
