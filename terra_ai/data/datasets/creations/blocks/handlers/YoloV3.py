from typing import Optional
from pydantic.types import PositiveInt

from terra_ai.data.datasets.extra import (
    LayerYoloChoice,
    LayerODDatasetTypeChoice,
    LayerObjectDetectionModelChoice,
    LayerImageFrameModeChoice,
)
from terra_ai.data.mixins import BaseMixinData


class OptionsData(BaseMixinData):
    # yolo: LayerYoloChoice = LayerYoloChoice.v4
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo_terra

    # Внутренние параметры
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
