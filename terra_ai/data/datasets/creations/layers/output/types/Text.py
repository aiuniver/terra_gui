from enum import Enum
from typing import Optional, List
from pydantic import validator, PositiveInt

from ...extra import ParametersBaseData


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
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    max_words: Optional[PositiveInt]
    put: Optional[str]
    deploy: Optional[bool] = False

    @validator("text_mode")
    def _validate_text_mode(cls, value: TextModeChoice) -> TextModeChoice:
        if value == TextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        else:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("word_to_vec")
    def _validate_word_to_vec(cls, value: bool) -> bool:
        if value:
            cls.__fields__["word_to_vec_size"].required = True
        return value
