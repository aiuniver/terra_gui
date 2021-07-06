from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException

from .base import BaseResponse


def handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, APIException):
        response = BaseResponse(
            status=response.status_code, error=response.data, success=False
        )
    return response
