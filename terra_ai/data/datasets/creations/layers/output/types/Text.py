from typing import Optional
from pydantic import validator
from pydantic.types import DirectoryPath, PositiveInt

from ......mixins import BaseMixinData
from .....extra import LayerPrepareMethodChoice


class ParametersData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: PositiveInt
    step: PositiveInt
    max_words_count: PositiveInt
    pymorphy: Optional[bool] = False
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    word_to_vec_size: Optional[PositiveInt]

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value
