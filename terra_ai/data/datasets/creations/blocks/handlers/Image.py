from pydantic import PositiveInt

from terra_ai.data.datasets.extra import LayerNetChoice, LayerImageFrameModeChoice
from terra_ai.data.datasets.creations.blocks.extra import ImageScalerData


class OptionsData(ImageScalerData):
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
