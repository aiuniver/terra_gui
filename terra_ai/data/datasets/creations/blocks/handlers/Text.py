from typing import Optional
from pydantic import validator, PositiveInt

from terra_ai.data.datasets.extra import LayerTextModeChoice
from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    text_mode: LayerTextModeChoice
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    pymorphy: bool
    filters: str
    # prepare_method: LayerPrepareMethodChoice
    # max_words_count: Optional[PositiveInt]
    # word_to_vec_size: Optional[PositiveInt]

    # Внутренние параметры
    deploy: Optional[bool] = False
    open_tags: Optional[str]
    close_tags: Optional[str]

    @validator("text_mode")
    def _validate_text_mode(cls, value):
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    # @validator("prepare_method")
    # def _validate_prepare_method(cls, value):
    #     if value in [
    #         LayerPrepareMethodChoice.embedding,
    #         LayerPrepareMethodChoice.bag_of_words,
    #     ]:
    #         cls.__fields__["max_words_count"].required = True
    #     elif value == LayerPrepareMethodChoice.word_to_vec:
    #         cls.__fields__["word_to_vec_size"].required = True
    #     return value
