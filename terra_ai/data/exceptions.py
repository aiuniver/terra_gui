from enum import Enum
from typing import Any


class ExceptionMessages(str, Enum):
    TerraDataValue = "%s: Value error"
    ValueType = "%s: Value must be a %s"
    AliasException = '%s: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
    UniqueListIdentifier = 'Identifier "%s" is undefined as attribute of %s'
    UniqueListUndefinedIdentifier = 'Identifier is undefined in "%s"'
    PartTotal = "%s: Sum of all properties must by 1"
    ListEmpty = "%s: must not be empty"
    FilePathExtension = '%s: File must have "%s" extension'
    Base64Extension = "Incorrect base64 string value"
    LayersNotInAvailable = "Layers %s not in available %s"
    LayersNotDefined = "You must define layers %s"
    PositionXY = "%s: Position must be a list with 2 elements, received %s"


class TerraDataException(ValueError):
    pass


class TerraDataValueException(TerraDataException):
    class Meta:
        message: str = ExceptionMessages.TerraDataValue

    def __init__(self, __value: Any, *args):
        super().__init__(((args[0] if len(args) else self.Meta.message) % str(__value)))


class ValueTypeException(TerraDataException):
    class Meta:
        message: str = ExceptionMessages.ValueType

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


class PartTotalException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.PartTotal


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


class FilePathExtensionException(TerraDataException):
    def __init__(self, __file_path: Any, __extension: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.FilePathExtension)
                % (str(__file_path), str(__extension))
            )
        )


class Base64Exception(TerraDataException):
    def __init__(self):
        super().__init__(ExceptionMessages.Base64Extension.value)


class LayersNotInAvailableException(TerraDataException):
    def __init__(self, __not_in_available: Any, __available: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.LayersNotInAvailable)
                % (str(__not_in_available), str(__available))
            )
        )


class LayersNotDefinedException(TerraDataException):
    def __init__(self, __not_defined: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.LayersNotDefined)
                % str(__not_defined)
            )
        )


class PositionXYException(TerraDataException):
    def __init__(self, __name: Any, __position: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.PositionXY)
                % (str(__name), str(__position))
            )
        )
