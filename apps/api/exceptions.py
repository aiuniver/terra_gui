import json
import logging
import traceback

from rest_framework.views import exception_handler
from rest_framework.exceptions import ValidationError

from apps.api.base import BaseResponseError
from apps.api.logging import LogData, LevelnameChoice


def handler(exc, context):
    response = exception_handler(exc, context)
    if isinstance(exc, ValidationError):
        title = "Ошибка валидации данных"
        message = json.dumps(exc.args, indent=2, ensure_ascii=False)
        logging.getLogger("django.request").error(traceback.format_exc())

    else:
        title = str(exc)
        message = None if response else traceback.format_exc()
        logging.getLogger("django.request").error(title if message is None else message)

    return BaseResponseError(
        LogData(level=LevelnameChoice.ERROR, title=title, message=message),
        data=getattr(exc, "data", None),
    )
