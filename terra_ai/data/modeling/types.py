import re

from ..exceptions import IncorrectReferenceNameException
from .extra import ReferenceTypeChoice


class ReferenceLayerType(str):
    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self, value: str) -> str:
        if not value:
            return value
        types = "|".join(ReferenceTypeChoice.values())
        match = re.match(rf"^({types})@[a-z]+[a-z0-9_]*$", value)
        if not match:
            raise IncorrectReferenceNameException(
                value, f"^({types})@[a-z]+[a-z0-9_]*$"
            )
        return str(value)
