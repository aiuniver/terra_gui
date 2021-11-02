from typing import Optional, List, Tuple, Any
from pydantic import validator
from pydantic.types import PositiveInt
from pydantic.errors import EnumMemberError

from .extra import BlockGroupChoice
from . import blocks
from ..mixins import BaseMixinData, IDMixinData, UniqueListMixin


class BlockBindData(BaseMixinData):
    up: List[PositiveInt] = []
    down: List[PositiveInt] = []


class BlockData(IDMixinData):
    name: str
    group: BlockGroupChoice
    bind: BlockBindData = BlockBindData()
    position: Tuple[int, int]
    parameters: Any

    @validator("group", pre=True)
    def _validate_group(cls, value: BlockGroupChoice) -> BlockGroupChoice:
        if value not in list(BlockGroupChoice):
            raise EnumMemberError(enum_values=list(BlockGroupChoice))
        name = (
            value if isinstance(value, BlockGroupChoice) else BlockGroupChoice(value)
        ).name
        type_ = getattr(blocks, getattr(blocks.Block, name))
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class BlocksList(UniqueListMixin):
    class Meta:
        source = BlockData
        identifier = "id"
