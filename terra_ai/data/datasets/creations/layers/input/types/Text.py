from enum import Enum
from typing import Optional, List
from pydantic import validator, PositiveInt

from ...extra import ParametersBaseData
from .....extra import LayerPrepareMethodChoice


class TextModeChoice(str, Enum):
    completely = "completely"
    length_and_step = "length_and_step"


class ParametersData(ParametersBaseData):
    cols_names: Optional[List[str]]
    max_words_count: PositiveInt
    pymorphy: Optional[bool] = False
    embedding: Optional[bool] = False
    bag_of_words: Optional[bool] = False
    word_to_vec: Optional[bool] = False
    word_to_vec_size: Optional[PositiveInt]
    delete_symbols: Optional[str]
    text_mode: TextModeChoice = TextModeChoice.completely
    max_words_count: PositiveInt
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    max_words: Optional[PositiveInt]
    put: Optional[str]

    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value

    @validator("text_mode", allow_reuse=True)
    def _validate_prepare_method(
            cls, value: TextModeChoice
    ) -> TextModeChoice:
        if value == TextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        else:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
