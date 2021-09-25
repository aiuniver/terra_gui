from enum import Enum

from .base import TerraBaseException


class ExceptionMessages(dict, Enum):
    Unknown = {'ru': '%s',
               'eng': '%s'}
    OperatorNotAllowedInGraph = {'ru': 'Недопустимый оператор. %s',
                                 'eng': 'Invalid operator. %s'}
    Cancelled = {'ru': 'Операция была отменена другой операцией. %s',
                 'eng': 'The operation is canceled by another operation. %s'}
    InvalidArgument = {'ru': 'Получен недопустимый аргумент. %s',
                       'eng': 'Invalid argument received. %s'}
    DeadlineExceeded = {'ru': 'Операция выполняется слишком долго. %s',
                        'eng': 'The operation is taking too long. %s'}
    NotFound = {'ru': 'Файл или каталог не найден. %s',
                'eng': 'File or directory was not found. %s'}
    AlreadyExists = {'ru': 'Объект уже существует. %s',
                     'eng': 'The object already exists. %s'}
    PermissionDenied = {'ru': 'Недостаточно прав для исполнения операции. %s',
                        'eng': 'Insufficient permissions to perform the operation. %s'}
    ResourceExhausted = {'ru': 'Недостаточно ресурсов системы для завершения операции. %s',
                         'eng': 'There are not enough system resources to complete the operation. %s'}
    FailedPrecondition = {'ru': 'Попытка использовать неинициализированное значение "tf.Variable". %s',
                          'eng': 'Attempting to use uninitialized value "tf.Variable". %s'}
    Aborted = {
     'ru': 'Возможно, старая сессия не закрыта либо текущую используют несколько клиентов. %s',
     'eng': 'It is possible that the old session is not closed, or several clients are using the current session. %s'}
    OutOfRange = {'ru': 'Итерация выходит за пределы допустимого диапазона входных данных. %s',
                  'eng': 'The operation iterates past the valid input range. %s'}
    Unimplemented = {'ru': 'Операция не поддерживается. %s',
                     'eng': 'The operation is not supported. %s'}
    Unavailable = {'ru': 'Среда выполнения в данный момент недоступна. %s',
                   'eng': 'The runtime is currently unavailable. %s'}
    DataLoss = {'ru': 'Данные утеряны. %s',
                'eng': 'Data is lost. %s'}


class TFBaseException(TerraBaseException):
    class Meta:
        message = ExceptionMessages.Unknown


class AbortedError(TFBaseException):
    class Meta:
        message = ExceptionMessages.Aborted


class AlreadyExistsError(TFBaseException):
    class Meta:
        message = ExceptionMessages.AlreadyExists


class CancelledError(TFBaseException):
    class Meta:
        message = ExceptionMessages.Cancelled


class DataLossError(TFBaseException):
    class Meta:
        message = ExceptionMessages.DataLoss


class DeadlineExceededError(TFBaseException):
    class Meta:
        message = ExceptionMessages.DeadlineExceeded


class FailedPreconditionError(TFBaseException):
    class Meta:
        message = ExceptionMessages.FailedPrecondition


class InvalidArgumentError(TFBaseException):
    class Meta:
        message = ExceptionMessages.InvalidArgument


class NotFoundError(TFBaseException):
    class Meta:
        message = ExceptionMessages.NotFound


class OperatorNotAllowedInGraphError(TFBaseException):
    class Meta:
        message = ExceptionMessages.OperatorNotAllowedInGraph


class OutOfRangeError(TFBaseException):
    class Meta:
        message = ExceptionMessages.OutOfRange


class PermissionDeniedError(TFBaseException):
    class Meta:
        message = ExceptionMessages.PermissionDenied


class ResourceExhaustedError(TFBaseException):
    class Meta:
        message = ExceptionMessages.ResourceExhausted


class UnavailableError(TFBaseException):
    class Meta:
        message = ExceptionMessages.Unavailable


class UnimplementedError(TFBaseException):
    class Meta:
        message = ExceptionMessages.Unimplemented


class UnknownError(TFBaseException):
    class Meta:
        message = ExceptionMessages.Unknown
