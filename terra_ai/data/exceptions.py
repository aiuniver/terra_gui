"""
Список исключений
"""

from enum import Enum
from typing import Any

from .. import settings


class ExceptionMessages(str, Enum):
    TerraDataValue = "%s: Value error"
    ValueType = "%s: Value must be a %s"
    AliasException = (
        '%s: It is allowed to use only lowercase latin characters, numbers and the "_" sign, '
        "must always begin with a latin character"
    )
    UniqueListIdentifier = 'Identifier "%s" is undefined as attribute of %s'
    UniqueListUndefinedIdentifier = 'Identifier is undefined in "%s"'
    PartTotal = "%s: Sum of all properties must by 1"
    ListEmpty = "%s: must not be empty"
    FilePathExtension = '%s: File name must have "%s" extension'
    DirectoryPathExtension = '%s: Directory name must have "%s" extension'
    FileNameExtension = '%s: File name must have "%s" extension'
    Base64Extension = "Incorrect base64 string value"
    ValueNotInList = "%s: Value must be in list %s"
    TrdsDirExt = (
        f"Dataset dirname must have `.{settings.DATASET_EXT}` extension, received `%s`"
    )
    TrdsConfigFileNotFound = "Dataset `%s` has not `%s`"
    LayerValueConfig = (
        "Validation method `%s` and value `%s` with type `%s` are mismatch"
    )
    IncorrectReferenceName = "Reference name `%s` is not value, must be `%s`"
    ObjectDetectionQuantityLayers = (
        "%s: При типе задачи `Обнаружение объектов` должен быть один выходной слой"
    )


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


class DirectoryPathExtensionException(TerraDataException):
    def __init__(self, __dir_path: Any, __extension: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.DirectoryPathExtension)
                % (str(__dir_path), str(__extension))
            )
        )


class FileNameExtensionException(TerraDataException):
    def __init__(self, __file_name: Any, __extension: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.FileNameExtension)
                % (str(__file_name), str(__extension))
            )
        )


class Base64Exception(TerraDataException):
    def __init__(self):
        super().__init__(ExceptionMessages.Base64Extension.value)


class ValueNotInListException(TerraDataException):
    def __init__(self, __value: Any, __items: Any, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.ValueNotInList)
                % (str(__value), str(__items))
            )
        )


class TrdsDirExtException(TerraDataException):
    def __init__(self, __dirname: str, *args):
        super().__init__(
            ((args[0] if len(args) else ExceptionMessages.TrdsDirExt) % str(__dirname))
        )


class TrdsConfigFileNotFoundException(TerraDataException):
    def __init__(self, __trds_name: str, __config_name: str, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.TrdsConfigFileNotFound)
                % (str(__trds_name), str(__config_name))
            )
        )


class LayerValueConfigException(TerraDataException):
    def __init__(self, __validator: str, __value: str, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.LayerValueConfig)
                % (str(__validator), str(__value), str(type(__value)))
            )
        )


class IncorrectReferenceNameException(TerraDataException):
    def __init__(self, __value: str, __regex: str, *args):
        super().__init__(
            (
                (args[0] if len(args) else ExceptionMessages.IncorrectReferenceName)
                % (str(__value), str(__regex))
            )
        )


class ObjectDetectionQuantityLayersException(TerraDataValueException):
    class Meta:
        message: str = ExceptionMessages.ObjectDetectionQuantityLayers
