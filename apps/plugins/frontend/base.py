from typing import Any, Optional, Union, List, Dict
from pydantic import validator

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
    label: Optional[str]
    placeholder: Optional[str]
    parse: Optional[str]
    value: Any = ""
    disabled: bool = False
    visible: bool = True
    changeable: bool = False
    list: Optional[List[Union[ListOptionData, ListOptgroupData]]]
    fields: Optional[Dict[str, List]]
    manual: Optional[Dict[str, str]]
    api: Optional[str]

    @validator("fields", always=True)
    def _validate_fields(
        cls, value: Optional[Dict[str, List]]
    ) -> Optional[Dict[str, List]]:
        if not value:
            return value
        for name, fields in value.items():
            __fields = []
            for field in fields:
                __fields.append(Field(**field))
            value.update({name: __fields})
        return value
