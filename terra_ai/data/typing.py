"""
## Типы полей
"""

import re
import base64
import binascii

from typing import Type
from pydantic import FilePath

from .exceptions import AliasException, Base64Exception, FilePathExtensionException


class AliasType(str):
    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self, value: str) -> str:
        if not re.match("^[a-z]+[a-z0-9_]*$", value):
            raise AliasException(value)
        return str(value)


class Base64Type(str):
    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self, value: str) -> str:
        try:
            base64.b64decode(value)
        except binascii.Error:
            raise Base64Exception()
        return str(value)


class FilePathType(FilePath):
    ext: str

    @classmethod
    def validate(self, value: FilePath) -> FilePath:
        value = super().validate(value)
        if f".{self.ext}" != value.suffix:
            raise FilePathExtensionException(value, self.ext)
        return value


def confilepath(*, ext: str) -> Type[FilePath]:
    namespace = dict(ext=ext)
    return type("FilePathType", (FilePathType,), namespace)
