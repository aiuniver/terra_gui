from enum import Enum
from typing import Optional
from pydantic import validator
from pydantic.types import DirectoryPath, PositiveInt

from ...extra import FileInfo
from ......mixins import BaseMixinData
from .....extra import LayerPrepareMethodChoice


class TextModeChoice(str, Enum):
    completely = 'Целиком'
    length_and_step = 'По длине и шагу'


class ParametersData(BaseMixinData):
    file_info: FileInfo
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

    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    # Разобраться с Артуром по
    #     pymorphy: Optional[bool] = False
    #     embedding: Optional[bool] = False
    #     bag_of_words: Optional[bool] = False
    #     word_to_vec: Optional[bool] = False
    #     image_path

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value
