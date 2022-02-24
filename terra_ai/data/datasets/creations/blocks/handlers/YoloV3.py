from typing import Optional
from pydantic.types import PositiveInt

from terra_ai.data.datasets.extra import (
    LayerODDatasetTypeChoice,
    LayerObjectDetectionModelChoice,
    LayerImageFrameModeChoice,
)
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    model_type: LayerODDatasetTypeChoice

    # Внутренние параметры
    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
