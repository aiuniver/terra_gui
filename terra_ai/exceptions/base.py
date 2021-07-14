from enum import Enum
from typing import Any


class TerraBaseMessages(str, Enum):
    Value = "Incorrect value '%s'"


class TerraBaseException(Exception):
    class Meta:
        message: str = TerraBaseMessages.Value

    def __init__(self, value: Any, *args):
        super().__init__(((args[0] if len(args) else self.Meta.message) % str(value)))
