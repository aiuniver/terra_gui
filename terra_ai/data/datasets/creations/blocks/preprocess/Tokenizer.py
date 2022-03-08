from pydantic.types import PositiveInt
from typing import Optional
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData
from terra_ai.data.datasets.extra import LayerPrepareMethodChoice


class OptionsData(BaseOptionsData):
    prepare_method: LayerPrepareMethodChoice
    max_words_count: PositiveInt
    lower: bool
    char_level: bool
    # Внутренние параметры
    filters: Optional[str]
    length: Optional[PositiveInt]
