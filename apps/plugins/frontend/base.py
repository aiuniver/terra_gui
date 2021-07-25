from typing import Any, Optional, Union, List

from terra_ai.data.types import AliasType
from terra_ai.data.mixins import BaseMixinData

from .extra import FieldTypeChoice


class ListOptionData(BaseMixinData):
    value: str
    label: str


class ListOptgroupData(BaseMixinData):
    label: str
    items: List[ListOptionData]


class Field(BaseMixinData):
    type: FieldTypeChoice
    name: AliasType
    label: str
    parse: str
    value: Any = ""
    list: Optional[List[Union[ListOptionData, ListOptgroupData]]]
