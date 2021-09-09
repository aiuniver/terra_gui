from enum import Enum

from .base import TerraBaseException


class TFMessages(str, Enum):
    Aborted = "Операция была прервана из-за одновременного действия"
    AlreadyExists = "объект, который мы пытались создать, уже существует"
    Cancelled = "отмена операции или шага"
    DataLoss = "безвозвратная потеря или повреждение данных"
    DeadlineExceeded = "истек крайний срок до завершения операции"
    FailedPrecondition = "Операция отклонена, потому что система не в состоянии ее выполнить"
    Internal = "в системе возникла внутренняя ошибка"
    InvalidArgument = "операция получила недопустимый аргумент"
    NotFound = "когда запрошенный объект (например, файл или каталог) не найден"
    OperatorNotAllowedInGraph = "ошибка для неподдерживаемого оператора при выполнении Graph"
    OutOfRange = "операция итерация выходит за пределы допустимого диапазона ввода"
    PermissionDenied = "у вызывающего абонента нет разрешения на выполнение операции"
    ResourceExhausted = "Некоторые ресурсы исчерпаны"
    Unauthenticated = " Запрос не имеет действительных учетных данных для аутентификации"
    Unavailable = "среда выполнения в данный момент недоступна"
    Unimplemented = "операция не была реализована"
    Unknown = "Неизвестная ошибка"


class TFException(TerraBaseException):
    class Meta:
        message: str = "Undefined error of TensorFlow"


class TFAbortedException(TFException):
    class Meta:
        message: str = TFMessages.Aborted


class TFAlreadyExistsException(TFException):
    class Meta:
        message: str = TFMessages.AlreadyExists


class TFCancelledException(TFException):
    class Meta:
        message: str = TFMessages.Cancelled


class TFDataLossException(TFException):
    class Meta:
        message: str = TFMessages.DataLoss


class TFDeadlineExceededException(TFException):
    class Meta:
        message: str = TFMessages.DeadlineExceeded


class TFFailedPreconditionException(TFException):
    class Meta:
        message: str = TFMessages.FailedPrecondition


class TFInternalException(TFException):
    class Meta:
        message: str = TFMessages.Internal


class TFInvalidArgumentException(TFException):
    class Meta:
        message: str = TFMessages.InvalidArgument


class TFNotFoundException(TFException):
    class Meta:
        message: str = TFMessages.NotFound


class TFOperatorNotAllowedInGraphException(TFException):
    class Meta:
        message: str = TFMessages.OperatorNotAllowedInGraph


class TFOutOfRangeException(TFException):
    class Meta:
        message: str = TFMessages.OutOfRange


class TFPermissionDeniedException(TFException):
    class Meta:
        message: str = TFMessages.PermissionDenied


class TFResourceExhaustedException(TFException):
    class Meta:
        message: str = TFMessages.ResourceExhausted


class TFUnauthenticatedException(TFException):
    class Meta:
        message: str = TFMessages.Unauthenticated


class TFUnavailableException(TFException):
    class Meta:
        message: str = TFMessages.Unavailable


class TFUnimplementedException(TFException):
    class Meta:
        message: str = TFMessages.Unimplemented


class TFUnknownException(TFException):
    class Meta:
        message: str = TFMessages.Unknown
