from enum import Enum
from typing import Any


class ExceptionMessages(str, Enum):
    TerraDataValue = "%s: Value error"
    AliasException = '%s: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
    UniqueListIdentifier = 'Identifier "%s" is undefined as attribute of %s'
    UniqueListUndefinedIdentifier = 'Identifier is undefined in "%s"'
    FilePathExtension = '%s: File must have "%s" extension'


class DataException(ValueError):
    pass


class DataValueException(DataException):
    class Meta:
        message: str = ExceptionMessages.TerraDataValue

    def __init__(self, __value: Any, *args):
        super().__init__(((args[0] if len(args) else self.Meta.message) % str(__value)))


class AliasException(DataValueException):
    class Meta:
        message: str = ExceptionMessages.AliasException


class UniqueListIdentifierException(DataException):
    def __init__(self, __identifier: Any, __source: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.UniqueListIdentifier)
                % (str(__identifier), str(__source))
            )
        )


class UniqueListUndefinedIdentifierException(DataException):
    def __init__(self, __source: Any, *args):
        super().__init__(
            (
                (
                    args[0]
                    if len(args)
                    else ExceptionMessages.UniqueListUndefinedIdentifier
                )
                % str(__source)
            )
        )


class FilePathExtensionException(DataException):
    def __init__(self, __file_path: Any, __extension: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.FilePathExtension)
                % (str(__file_path), str(__extension))
            )
        )
