from typing import Optional

from ......mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerTextModeChoice
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
