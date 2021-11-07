from typing import Optional

from ......mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTextModeChoice
from pydantic import validator
from pydantic.types import PositiveInt


class ParametersData(BaseMixinData):
    open_tags: Optional[str]
    close_tags: Optional[str]

    sources_paths: Optional[list]
    filters: Optional[str]
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    put: Optional[PositiveInt]
    cols_names: Optional[str]

    def __init__(self, **data):
        data.update({"cols_names": None})
        super().__init__(**data)

    @validator("text_mode")
    def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value
