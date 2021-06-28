from enum import Enum
from typing import Any


class ExceptionMessages(str, Enum):
    AliasException = '%s: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
    UniqueListIdentifier = 'Identifier "%s" is undefined as attribute of %s'
    UniqueListUndefinedIdentifier = 'Identifier is undefined in "%s"'
    PositiveInteger = "%s: Value must be greater or equivalent then 1"
    PartValue = "%s: Value must be between 0 and 1"
    PartTotal = "%s: Sum of all properties must by 1"
    ZipFile = "%s: Value must be a zip-file"
    ListEmpty = "%s must not be empty"


class TerraDataException(ValueError):
    pass


class TerraDataValueException(TerraDataException):
    class Meta:
        message: str = "%s: Value error"

    def __init__(self, __value: Any, *args):
        super().__init__(((args[0] if len(args) else self.Meta.message) % str(__value)))


class ValueTypeException(TerraDataException):
    class Meta:
        message: str = "%s: Value must be a %s"

    def __init__(self, __value: Any, __type: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else self.Meta.message)
                % (str(__value), str(__type))
            )
        )


class AliasException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.AliasException


class PositiveIntegerException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.PositiveInteger


class PartValueException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.PartValue


class PartTotalException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.PartTotal


class ZipFileException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.ZipFile


class ListEmptyException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.ListEmpty


class UniqueListIdentifierException(TerraDataException):
    def __init__(self, __identifier: Any, __source: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.UniqueListIdentifier)
                % (str(__identifier), str(__source))
            )
        )


class UniqueListUndefinedIdentifierException(TerraDataException):
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
