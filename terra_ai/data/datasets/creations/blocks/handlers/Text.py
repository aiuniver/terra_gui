from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
)


class OptionsData(BaseMixinData):
    text_mode: LayerTextModeChoice
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    prepare_method: LayerPrepareMethodChoice
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]
    pymorphy: bool  # = False
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'

    # Внутренние параметры
    deploy: Optional[bool] = False
    open_tags: Optional[str]
    close_tags: Optional[str]

    def __init__(self, **data):
        data.update({"cols_names": {}})
        super().__init__(**data)

    @validator("text_mode")
    def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("prepare_method")
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value in [
            LayerPrepareMethodChoice.embedding,
            LayerPrepareMethodChoice.bag_of_words,
        ]:
            cls.__fields__["max_words_count"].required = True
        elif value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value
