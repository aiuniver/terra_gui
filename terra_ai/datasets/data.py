"""
Datasets constants data
"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic.color import Color
from pydantic.types import PositiveInt

from terra_ai.data.mixins import BaseMixinData, UniqueListMixin


DataType = {0: "DIM", 1: "DIM", 2: "1D", 3: "2D", 4: "3D", 5: "4D"}


class Preprocesses(str, Enum):
    scaler = "scaler"
    tokenizer = "tokenizer"
    word2vec = "word2vec"
    augmentation = "augmentation"


class InstructionsData(BaseMixinData):
    instructions: Any  # [Union[str, PositiveInt, Dict[str, List[Union[StrictIntValueGe0, StrictFloatValueGe0]]]]]
    parameters: Any

    # instructions: Dict[str, Union[str, StrictIntValueGe0, StrictFloatValueGe0]]

    # @validator("parameters", always=True)
    # def _validate_parameters(cls, value: Any, values, field) -> Any:
    #     return field.type_(**value or {})


class DatasetInstructionsData(BaseMixinData):
    inputs: Dict[PositiveInt, Dict[Any, InstructionsData]]
    outputs: Dict[PositiveInt, Dict[Any, InstructionsData]]
    # service: Optional[Dict[PositiveInt, Dict[Any, InstructionsData]]]


class ColorHex(Color):
    def __str__(self) -> str:
        return self.as_hex()


class AnnotationClassData(BaseMixinData):
    name: str
    color: ColorHex


class AnnotationClassesList(UniqueListMixin):
    class Meta:
        source = AnnotationClassData
        identifier = "name"

    @property
    def colors_as_rgb_list(self) -> list:
        return list(map(lambda item: item.color.as_rgb_tuple(), self))
