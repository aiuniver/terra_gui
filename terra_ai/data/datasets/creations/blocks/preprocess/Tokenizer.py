from pydantic.types import PositiveInt
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    max_words_count: PositiveInt
    filters: str  # = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    lower: bool
    char_level: bool
