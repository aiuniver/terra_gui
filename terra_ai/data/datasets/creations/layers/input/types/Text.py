from enum import Enum
from typing import Optional, List, Union
from pydantic import validator, FilePath, DirectoryPath, PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerPrepareMethodChoice


class TextModeChoice(str, Enum):
    completely = 'Целиком'
    length_and_step = 'По длине и шагу'


class ParametersData(BaseMixinData):
    sources_paths: List[Union[DirectoryPath, FilePath]]
    cols_names: Optional[List[str]]
    pymorphy: Optional[bool] = False
    embedding: Optional[bool] = False
    bag_of_words: Optional[bool] = False
    word_to_vec: Optional[bool] = False
    word_to_vec_size: Optional[PositiveInt]
    delete_symbols: Optional[str]
    text_mode: TextModeChoice = TextModeChoice.completely
    length: PositiveInt
    step: PositiveInt
    max_words: PositiveInt
    put: Optional[str]

    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding # ???

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value
