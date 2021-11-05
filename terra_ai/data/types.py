"""
## Типы полей
"""

import re
import base64
import binascii

from typing import Optional, Type
from pydantic import FilePath
from pydantic.types import conint, confloat, constr, PositiveInt, DirectoryPath

from .exceptions import (
    AliasException,
    Base64Exception,
    FilePathExtensionException,
    FileNameExtensionException,
    DirectoryPathExtensionException,
)


ConstrainedIntValueGe0 = conint(ge=0)
ConstrainedIntValueGe1 = conint(ge=1)
ConstrainedIntValueGe2 = conint(ge=2)
ConstrainedIntValueGe0Le2 = conint(ge=0, le=2)
StrictIntValueGe0 = conint(strict=True, ge=0)

ConstrainedFloatValueGe0 = confloat(ge=0)
ConstrainedFloatValueLe0 = confloat(le=0)
ConstrainedFloatValueGe0Le1 = confloat(ge=0, le=1)
ConstrainedFloatValueGe0Le100 = confloat(ge=0, le=100)
StrictFloatValueGe0 = confloat(strict=True, ge=0)


ConstrainedLayerNameValue = constr(max_length=32)


class IDType(PositiveInt):
    pass


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


class DirectoryPathType(DirectoryPath):
    ext: Optional[str] = None

    @classmethod
    def validate(self, value: DirectoryPath) -> DirectoryPath:
        value = super().validate(value)
        if self.ext is not None and f".{self.ext}" != value.suffix:
            raise DirectoryPathExtensionException(value, self.ext)
        return value


def confilepath(*, ext: str) -> Type[FilePath]:
    namespace = dict(ext=ext)
    return type("FilePathType", (FilePathType,), namespace)


def condirpath(*, ext: str = None) -> Type[DirectoryPath]:
    namespace = dict()
    if ext is not None:
        namespace = dict(ext=ext)
    
    return type("DirectoryPathType", (DirectoryPathType,), namespace)

class FileNameType(str):
    ext: str

    @classmethod
    def __get_validators__(self):
        yield self.validate

    @classmethod
    def validate(self, value: str) -> str:
        if not str(value).endswith(f".{self.ext}"):
            raise FileNameExtensionException(value, self.ext)
        return value


def confilename(*, ext: str) -> str:
    namespace = dict(ext=ext)
    return type("FileNameType", (FileNameType,), namespace)
