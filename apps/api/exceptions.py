import json
import subprocess
import traceback
from pathlib import Path

from django.conf import settings
from rest_framework.views import exception_handler
from rest_framework.exceptions import APIException

from . import base


def save_errors(func):
    def wrapper(exc, context):
        source = Path(settings.BASE_DIR, "errors.txt")
        with open(source, 'w') as file:
            view = 'View: ' + repr(context.get("view", "")) + '\n'
            request = 'Request: ' + repr(context.get("request", "")) + '\n'
            trace = traceback.format_exc()
            for line in (view, request, trace):
                file.write(line + '\n')
            source.chmod(0o777)
        return func(exc, context)

    return wrapper


@save_errors
def handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, APIException):
        response = base.BaseResponseErrorGeneral(response.data)
    elif isinstance(context.get("view"), base.BaseAPIView):
        response = base.BaseResponseErrorGeneral(str(exc))
    return response
