from enum import Enum

from terra_ai.settings import LANGUAGE


class Messages(dict, Enum):
    Unknown = {"ru": "%s", "eng": "%s"}


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
