from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException

from .base import BaseResponseErrorGeneral, BaseAPIView


def handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, APIException):
        response = BaseResponseErrorGeneral(response.data)
    elif isinstance(context.get("view"), BaseAPIView):
        response = BaseResponseErrorGeneral(str(exc))
    return response
