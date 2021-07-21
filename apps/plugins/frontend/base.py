from typing import Any, Optional, Union, List, Dict
from pydantic import BaseModel

from .extra import FieldTypeChoice


class Field(BaseModel):
    type: FieldTypeChoice
    label: str
    default: Any
    disabled: bool = False
    readonly: bool = False
    list: bool = False
    available: Optional[Union[List, Dict]]
    available_names: Optional[Union[List, Dict]]
