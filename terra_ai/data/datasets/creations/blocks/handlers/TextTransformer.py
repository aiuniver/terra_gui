from typing import Optional
from pydantic import PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTextModeChoice, LayerPrepareMethodChoice


class OptionsData(BaseMixinData):

    # Внутренние параметры
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    prepare_method: Optional[LayerPrepareMethodChoice]
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]
    pymorphy: Optional[bool]
    filters: Optional[str]
