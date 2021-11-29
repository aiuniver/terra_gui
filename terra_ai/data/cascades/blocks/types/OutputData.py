from typing import Optional
from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerInputTypeChoice


class ParametersMainData(BaseMixinData):
    type: LayerInputTypeChoice = LayerInputTypeChoice.Video
    width: Optional[PositiveInt] = 640
    height: Optional[PositiveInt] = 480
