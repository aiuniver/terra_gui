from typing import Optional
from pydantic import PositiveInt

from terra_ai.data.datasets.extra import LayerTextModeChoice, LayerPrepareMethodChoice
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    open_tags: Optional[str]
    close_tags: Optional[str]

    # Внутренние параметры
    # prepare_method: Optional[LayerPrepareMethodChoice]
    filters: Optional[str]
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
