from pydantic import PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData  # , ImageScalerData
from terra_ai.data.datasets.extra import LayerNetChoice, LayerImageFrameModeChoice


class OptionsData(BaseOptionsData):  # ImageScalerData
    height: PositiveInt
    width: PositiveInt
    net: LayerNetChoice
    image_mode: LayerImageFrameModeChoice
