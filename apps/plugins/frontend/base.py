from typing import Optional, List
from pydantic import BaseModel

from .extra import WidgetTypeChoice
from .types import AliasType


class FieldBase(BaseModel):
    name: AliasType
    label: str
    type: WidgetTypeChoice
    parse: Optional[str]
    value: List[str] = [""]
    list: Optional[List[str]]
