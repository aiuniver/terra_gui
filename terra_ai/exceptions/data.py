import json
import os
from typing import Any, List, Sequence, Mapping, Dict

from pydantic import ValidationError, BaseModel, validator
from pydantic_i18n import PydanticI18n, BaseLoader

from terra_ai.exceptions.base import TerraBaseException
from terra_ai.assets.translations.extra import errors
from terra_ai.settings import TRANSLATIONS_DIR, LANGUAGE

LANGUAGE = "en_US" if LANGUAGE == "eng" else LANGUAGE


class JsonLoader(BaseLoader):
    def __init__(self, directory: str):
        if not os.path.exists(directory):
            raise OSError(f"Directory '{directory}' doesn't exist.")
        if not os.path.isdir(directory):
            raise OSError(f"'{directory}' is not a directory.")

        self.directory = directory

    @property
    def locales(self) -> Sequence[str]:
        locales: Sequence[str] = [
            filename[:-5]
            for filename in os.listdir(self.directory)
            if filename.endswith(".json")
        ]
        return locales

    def get_translations(self, locale: str) -> Mapping[str, str]:
        with open(os.path.join(self.directory, f"{locale}.json"), encoding='utf-8') as fp:
            data: Dict[str, str] = json.load(fp)

        return data


class Error(BaseModel):
    loc: Any
    msg: str
    err_type: str
    model: Any

    @validator("msg")
    def _validate_msg(cls, value: str):
        first_part = value.split(":")[0]
        if first_part in errors[LANGUAGE].keys():
            last_part = value.split(":")[1]
            value = f"{errors[LANGUAGE][first_part]}:{last_part}"
        return value


class PydanticException(TerraBaseException):
    errors: List[Error] = []

    def __init__(self, exception: ValidationError):
        if not isinstance(exception, ValidationError):
            raise TypeError(f"DataException ожидает на вход объект исключения ValidationError, '"
                            f"'но получил '{type(exception).__name__}'")
        loader = JsonLoader(TRANSLATIONS_DIR)
        translated_errors = PydanticI18n(loader).translate(exception.errors(), locale=LANGUAGE)
        self.errors = [Error(
            loc=error['loc'],
            msg=error['msg'],
            err_type=error['type'],
            model=exception.model)
            for error in translated_errors]
        super().__init__(str(self))

    def __str__(self):
        return "; ".join([str(error) for error in self.errors])
