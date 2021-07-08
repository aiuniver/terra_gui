import re

from typing import Type
from pydantic import FilePath

from .exceptions import AliasException, FilePathExtensionException


class AliasType(str):
    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self, value: str) -> str:
        if not re.match("^[a-z]+[a-z0-9_]*$", value):
            raise AliasException(value)
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
