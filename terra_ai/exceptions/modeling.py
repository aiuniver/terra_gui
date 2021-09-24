from enum import Enum

from .base import TerraBaseException


class ModelingMessages(dict, Enum):
    Undefined = {"ru": "Неопределенная ошибка моделирования",
                 "eng": "Undefined error of modeling"}
    # Connection
    LayerNotConnectedToMainPart = {"ru": "Слой не подключен к основной части",
                                   "eng": "Layer is not connected to main part"}
    # Input Shape
    ExpectedOtherInputShapeDim = {"ru": "Expected input shape dim=%s for %s but received dim=%s with input_shape `%s`",
                                  "eng": "Expected input shape dim=%s for %s but received dim=%s with input_shape `%s`"}
    InputShapeMustBeInEchDim = {"ru": "Input shape must be %s %s in each dim but received input shape %s",
                                "eng": "Input shape must be %s %s in each dim but received input shape %s"}
    InputShapeMustBeOnly = {"ru": "With %s input shape must be only %s but received: %s",
                            "eng": "With %s input shape must be only %s but received: %s"}
    InputShapeMustBeWholeDividedBy = {"ru": "Input shape `%s` except channels must be whole divided by %s",
                                      "eng": "Input shape `%s` except channels must be whole divided by %s"}
    LayerDoesNotHaveInputShape = {"ru": "Layer does not have input shape",
                                  "eng": "Layer does not have input shape"}
    # Output Shape
    UnexpectedOutputShape = {"ru": "На выходных данных ожидалась размерность %s, но получена %s",
                             "eng": "Expected output shape %s but got output shape %s"}
    UnspecifiedOutputLayer = {"ru": "Выходной слой не указан",
                              "eng": "Unspecified output layer"}
    # Parameters
    BadParameters = {"ru": "Проверьте следующие параметры: %s",
                     "eng": "Check the following parameters: %s"}
    InitializerCanTakeOnlyNDInputShape = {"ru": "%s initializer in %s can take only %sD input shape but "
                                                "received %sD input shape: %s",
                                          "eng": "%s initializer in %s can take only %sD input shape but "
                                                 "received %sD input shape: %s"}
    CanTakeOneOfTheFollowingValues = {"ru": "%s can take one of the following values: %s",
                                      "eng": "%s can take one of the following values: %s"}
    CannotHaveValue = {"ru": "%s cannot have value %s at the same time",
                       "eng": "%s cannot have value %s at the same time"}
    ClassesShouldBe = {"ru": "If %s, `classes` should be %s but received: %s",
                       "eng": "If %s, `classes` should be %s but received: %s"}
    DimensionSizeMustBeEvenlyDivisible = {"ru": "Dimension size (%s) from %s both must be evenly divisible by %s",
                                          "eng": "Dimension size (%s) from %s both must be evenly divisible by %s"}
    InputDimMustBeThenSizeOf = {"ru": "input_dim=%s must be %s then size of %s",
                                "eng": "input_dim=%s must be %s then size of %s"}
    ParameterCanNotBeForInputShape = {"ru": "For input shape with %s parameter %s can not be %s but received %s",
                                      "eng": "For input shape with %s parameter %s can not be %s but received %s"}
    ActivationFunctionShouldBe = {"ru": "If %s, activation function should be %s",
                                  "eng": "If %s, activation function should be %s"}
    # Position
    IncorrectNumberOfFiltersAndChannels = {"ru": "The number of filters %s and channels %s "
                                                 "must be evenly divisible by the number of groups %s",
                                           "eng": "The number of filters %s and channels %s "
                                                  "must be evenly divisible by the number of groups %s"}
    IncorrectQuantityInputShape = {"ru": "Expected %s input shape%s but got %s",
                                   "eng": "Expected %s input shape%s but got %s"}
    InputShapeEmpty = {"ru": "Получена пустая размерность входных данных",
                       "eng": "Received empty input shape"}
    # Input dimension
    IncorrectQuantityInputDimensions = {"ru": "Ожидаемое количество входов: %s, однако принято: %s",
                                        "eng": "Expected %s input dimensions but got %s"}
    InputShapesAreDifferent = {"ru": "Все размерности входных данных должны быть одинаковыми: %s",
                               "eng": "All input shapes must be the same but received: %s"}
    InputShapesHaveDifferentSizes = {"ru": "Во входной размерности присутствуют неодинаковые размеры: %s",
                                     "eng": "Input shapes have different sizes: %s"}
    MismatchedInputShapes = {
        "ru": "Required inputs with matching shapes except for the concat axis `%s` but received: %s",
        "eng": "Required inputs with matching shapes except for the concat axis `%s` but received: %s"}


class ModelingException(TerraBaseException):
    class Meta:
        message: dict = ModelingMessages.Undefined


class LayerNotConnectedToMainPartException(ModelingException):
    class Meta:
        message = ModelingMessages.LayerNotConnectedToMainPart


class ExpectedOtherInputShapeDimException(ModelingException):
    class Meta:
        message = ModelingMessages.ExpectedOtherInputShapeDim

    def __init__(self, __exp_dim, __exp_inp_shape, __rec_dim, __rec_inp_shape, **kwargs):
        super().__init__(str(__exp_dim), str(__exp_inp_shape), str(__rec_dim), str(__rec_inp_shape), **kwargs)


class InputShapeMustBeInEchDimException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapeMustBeInEchDim


class InputShapeMustBeOnlyException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapeMustBeOnly


class InputShapeMustBeWholeDividedByException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapeMustBeWholeDividedBy

    def __init__(self, __inp_shape, __num: int, **kwargs):
        super().__init__(str(__inp_shape), str(__num), **kwargs)


class LayerDoesNotHaveInputShapeException(ModelingException):
    class Meta:
        message = ModelingMessages.LayerDoesNotHaveInputShape

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class UnexpectedOutputShapeException(ModelingException):
    class Meta:
        message = ModelingMessages.UnexpectedOutputShape

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class UnspecifiedOutputLayerException(ModelingException):
    class Meta:
        message = ModelingMessages.UnspecifiedOutputLayer


class BadParametersException(ModelingException):
    class Meta:
        message = ModelingMessages.BadParameters

    def __init__(self, __params, **kwargs):
        super().__init__(str(__params), **kwargs)


class InitializerCanTakeOnlyNDInputShapeException(ModelingException):
    class Meta:
        message = ModelingMessages.InitializerCanTakeOnlyNDInputShape


class CanTakeOneOfTheFollowingValuesException(ModelingException):
    class Meta:
        message = ModelingMessages.CanTakeOneOfTheFollowingValues

    def __init__(self, __name, __values, **kwargs):
        super().__init__(str(__name), str(__values), **kwargs)


class CannotHaveValueException(ModelingException):
    class Meta:
        message = ModelingMessages.CannotHaveValue

    def __init__(self, __name, __value, **kwargs):
        super().__init__(str(__name), str(__value), **kwargs)


class ClassesShouldBeException(ModelingException):
    class Meta:
        message = ModelingMessages.ClassesShouldBe


class DimensionSizeMustBeEvenlyDivisibleException(ModelingException):
    class Meta:
        message = ModelingMessages.DimensionSizeMustBeEvenlyDivisible


class InputDimMustBeThenSizeOfException(ModelingException):
    class Meta:
        message = ModelingMessages.InputDimMustBeThenSizeOf


class ParameterCanNotBeForInputShapeException(ModelingException):
    class Meta:
        message = ModelingMessages.ParameterCanNotBeForInputShape


class ActivationFunctionShouldBeException(ModelingException):
    class Meta:
        message = ModelingMessages.ActivationFunctionShouldBe


class IncorrectNumberOfFiltersAndChannelsException(ModelingException):
    class Meta:
        message = ModelingMessages.IncorrectNumberOfFiltersAndChannels

    def __init__(self, __filters, __channels, __groups, **kwargs):
        super().__init__(str(__filters), str(__channels), str(__groups), **kwargs)


class IncorrectQuantityInputShapeException(ModelingException):
    class Meta:
        message = ModelingMessages.IncorrectQuantityInputShape

    def __init__(self, __expected, __suffix, __got, **kwargs):
        super().__init__(str(__expected), str(__suffix), str(__got), **kwargs)


class InputShapeEmptyException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapeEmpty


class IncorrectQuantityInputDimensionsException(ModelingException):
    class Meta:
        message = ModelingMessages.IncorrectQuantityInputDimensions

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)


class InputShapesAreDifferentException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapesAreDifferent


class InputShapesHaveDifferentSizesException(ModelingException):
    class Meta:
        message = ModelingMessages.InputShapesHaveDifferentSizes

    def __init__(self, __inp_shape, **kwargs):
        super().__init__(str(__inp_shape), **kwargs)


class MismatchedInputShapesException(ModelingException):
    class Meta:
        message = ModelingMessages.MismatchedInputShapes

    def __init__(self, __expected, __got, **kwargs):
        super().__init__(str(__expected), str(__got), **kwargs)
