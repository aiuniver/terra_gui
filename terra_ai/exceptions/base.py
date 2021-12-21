from enum import Enum

from terra_ai.settings import LANGUAGE


class Messages(dict, Enum):
    Unknown = {"ru": "%s", "eng": "%s"}
    NotDescribed = {
        "ru": "Еще не описанная ошибка. Класс `%s`, метод `%s`.",
        "eng": "An error not yet described. `%s` class, `%s` method.",
    }


class TerraBaseException(Exception):
    class Meta:
        message = Messages.Unknown

    def __init__(self, *args, lang: str = LANGUAGE):
        error_msg = self.Meta.message.value.get(lang)

        if args:
            try:
                error_msg = error_msg % args
            except TypeError:
                pass

        super().__init__(error_msg)


class NotDescribedException(TerraBaseException):
    class Meta:
        message: dict = Messages.NotDescribed

    def __init__(self, __module: str, __method: str, **kwargs):
        super().__init__(str(__module), str(__method), **kwargs)
