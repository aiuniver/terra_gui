from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException

from . import base


def handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, APIException):
        response = base.BaseResponseErrorGeneral(response.data)
    elif isinstance(context.get("view"), base.BaseAPIView):
        response = base.BaseResponseErrorGeneral(str(exc))
    return response
