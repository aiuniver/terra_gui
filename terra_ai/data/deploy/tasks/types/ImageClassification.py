from typing import List, Tuple
from pydantic import FilePath, PositiveFloat

from terra_ai.data.mixins import BaseMixinData


class Data(BaseMixinData):
    source: FilePath
    actual: str
    data: List[Tuple[str, PositiveFloat]]
