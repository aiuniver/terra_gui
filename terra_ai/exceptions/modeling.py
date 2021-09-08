from enum import Enum

from .base import TerraBaseException


class ModelingMessages(str, Enum):
    # Connection
    LayerNotConnectedToMainPart = "Layer is not connected to main part"
    # Input Shape
    ExpectedOtherInputShapeDim = "Expected input shape dim=%s for %s but received dim=%s with input_shape `%s`"
    InputShapeMustBeInEchDim = "Input shape must be %s %s in each dim but received input shape %s"
    InputShapeMustBeOnly = "With %s input shape must be only %s but received: %s"
    InputShapeMustBeWholeDividedBy = "Input shape `%s` except channels must be whole divided by %s"
    LayerHaveNotInputShape = "Layer does not have input shape"
    # Output Shape
    UnexpectedOutputShape = "Expected output shape `%s` but got output shape `%s`"
    UnspecifiedOutputLayer = "Unspecified output layer"
    # Parameters
    CheckFollowingParameters = "Check the following parameters"
    InitializerCanTakeOnlyNDInputShape = "%s initializer in %s can take only %sD input shape but "\
                                         "received %sD input shape: %s"
    CanTakeOneOfTheFollowingValues = "%s can take one of the following values %s"
    CannotHaveValue = "%s cannot have value %s at the same time"
    ClassesShouldBe = "If %s, `classes` should be %s but received: %s"
    DimensionSizeMustBeEvenlyDivisible = "Dimension size (%s) from %s both must be evenly divisible by %s"
    InputDimMustBeThenSizeOf = "input_dim=%s must be %s then size of %s"
    ParameterCanNotBeForInputShape = "For input shape with %s parameter %s can not be %s but received %s"
    # Position
    IncorrectNumberOfFiltersAndChannels = "The number of filters %s and channels %s " \
                                          "must be evenly divisible by the number of groups %s"
    IncorrectQuantityInputShape = "Expected %s input shape%s but got %s"
    InputShapeEmpty = "Received empty input shape"
    # Input dimension
    IncorrectQuantityInputDimensions = "Expected %s input dimensions but got %s"
    InputShapesAreDifferent = "All input shapes must be the same but received: %s"
    InputShapesHaveDifferentSizes = "Input shapes have different sizes: %s"
    MismatchedInputShapes = "Required inputs with matching shapes except for the concat axis `%s` but received: %s"


class ModelingException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of modeling"
