from typing import Optional
from pydantic import PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerImageFrameModeChoice,
)


class OptionsData(MinMaxScalerData, BaseMixinData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice
    scaler: LayerScalerImageChoice
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch

    # Внутренние параметры
    # object_detection: Optional[bool] = False
