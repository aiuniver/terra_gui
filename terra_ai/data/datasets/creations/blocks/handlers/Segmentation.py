from typing import List, Optional
from pydantic import PositiveInt
from pydantic.color import Color

from terra_ai.data.datasets.extra import LayerImageFrameModeChoice
from terra_ai.data.types import ConstrainedIntValueGe0
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    mask_range: ConstrainedIntValueGe0
    classes_names: List[str]
    classes_colors: List[Color]

    # Внутренние параметры
    width: Optional[PositiveInt]
    height: Optional[PositiveInt]
    image_mode: Optional[LayerImageFrameModeChoice]
