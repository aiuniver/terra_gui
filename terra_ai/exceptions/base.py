from enum import Enum

from .. import settings


class Messages(dict, Enum):
    Unknown = {"ru": "Неизвестная ошибка",
               "eng": "Undefined error"}


class TerraBaseException(Exception):
    class Meta:
        message = Messages.Unknown

    def __init__(self, *args, lang: str = settings.LANGUAGE):
        error_msg = self.Meta.message.value.get(lang)

        if args:
            try:
                error_msg = error_msg % args
            except TypeError:
                pass

        super().__init__(error_msg)
