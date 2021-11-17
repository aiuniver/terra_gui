from enum import Enum

from .base import TerraBaseException


class CascadesMessages(dict, Enum):
    Undefined = {"ru": "Неопределенная ошибка проектирования каскада",
                 "eng": "Undefined error of cascade project"}
    # Connection
    BlockNotConnectedToMainPart = {"ru": "Блок не подключен к основной части каскада или подключен неверно",
                                   "eng": "Block is not connected to main part of cascade"}

    # Parameters
    BadParameters = {"ru": "Проверьте следующие параметры: %s",
                     "eng": "Check the following parameters: %s"}

    CanTakeOneOfTheFollowingValues = {"ru": "%s can take one of the following values: %s",
                                      "eng": "%s can take one of the following values: %s"}

    # Input dimension
    IncorrectQuantityInputDimensions = {"ru": "Ожидаемое количество входов: %s, однако принято: %s",
                                        "eng": "Expected %s input dimensions but got %s"}

    # Data types
    DatasetDataDoesNotMatchInputData = {
        "ru": "Тип данных датасета: %s не соответствует типу данных входного блока: %s",
        "eng": "Dataset data type: %s does not match the input block data type: %s"
    }

    InputDataDoesNotMatchModelData = {
        "ru": "Тип данных входного блока: %s не соответствует типу данных модели: %s",
        "eng": "Input block data type: %s does not match the model data type: %s"
    }

    UsedDataDoesNotMatchBlockData = {
        "ru": "Тип данных блока: %s не соответствует типу используемых в каскаде данных: %s",
        "eng": "Block data type: %s does not match the type of data used in the cascade: %s"
    }

    BindCountNotEnough = {
        "ru": "Недостаточно входящих связей: необходимо %s. Возможны связи с блоками: %s",
        "eng": "Not enough incoming connections: %s is needed. Connections with blocks are possible: %s"
    }

    BindCountExceeding = {
        "ru": "Превышение количества входящих связей: необходимо %s. Возможны связи с блоками: %s",
        "eng": "Exceeding the number of incoming links: %s is needed. Connections with blocks are possible: %s"
    }

    ForbiddenBindExceeding = {
        "ru": "Не разрешенная связь с блоком %s. Возможны связи с блоками: %s",
        "eng": "Not allowed communication with the block %s. Connections with blocks are possible: %s"
    }

    BindInappropriateDataType = {
        "ru": "Тип данных блока %s не совпадает с типом данных каскада %s",
        "eng": "The data type of the %s block does not match the data type of the cascade %s"
    }


class CascadesException(TerraBaseException):
    class Meta:
        message: dict = CascadesMessages.Undefined


class BlockNotConnectedToMainPartException(CascadesException):
    class Meta:
        message = CascadesMessages.BlockNotConnectedToMainPart


class BadParametersException(CascadesException):
    class Meta:
        message = CascadesMessages.BadParameters

    def __init__(self, __params, **kwargs):
        super().__init__(str(__params), **kwargs)


class CanTakeOneOfTheFollowingValuesException(CascadesException):
    class Meta:
        message = CascadesMessages.CanTakeOneOfTheFollowingValues

    def __init__(self, __name, __values, **kwargs):
        super().__init__(str(__name), str(__values), **kwargs)


class IncorrectQuantityInputDimensionsException(CascadesException):
    class Meta:
        message = CascadesMessages.IncorrectQuantityInputDimensions

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class DatasetDataDoesNotMatchInputDataException(CascadesException):
    class Meta:
        message = CascadesMessages.DatasetDataDoesNotMatchInputData

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class InputDataDoesNotMatchModelDataException(CascadesException):
    class Meta:
        message = CascadesMessages.InputDataDoesNotMatchModelData

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class UsedDataDoesNotMatchBlockDataException(CascadesException):
    class Meta:
        message = CascadesMessages.UsedDataDoesNotMatchBlockData

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class BindCountNotEnoughException(CascadesException):
    class Meta:
        message = CascadesMessages.BindCountNotEnough

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class BindCountExceedingException(CascadesException):
    class Meta:
        message = CascadesMessages.BindCountExceeding

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class ForbiddenBindException(CascadesException):
    class Meta:
        message = CascadesMessages.ForbiddenBindExceeding

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class BindInappropriateDataTypeException(CascadesException):
    class Meta:
        message = CascadesMessages.BindInappropriateDataType

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)