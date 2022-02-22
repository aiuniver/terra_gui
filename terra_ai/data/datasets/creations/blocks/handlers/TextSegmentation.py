from typing import Optional
from pydantic import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTextModeChoice


class OptionsData(BaseMixinData):
    open_tags: Optional[str]
    close_tags: Optional[str]

    # Внутренние параметры
    filters: Optional[str]
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]