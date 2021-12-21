import re
import os

from uuid import uuid4
from pathlib import Path
from contextlib import contextmanager
from typing import Union, Tuple
from encodings.aliases import aliases as encodings_aliases

from .exceptions.base import TerraBaseException, NotDescribedException
from .settings import TMP_DIR


def decamelize(camel_case_string: str):
    string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    return re.sub("([a-z0-9])([A-Z0-9])", r"\1_\2", string).lower()


def camelize(snake_case_string: str):
    if snake_case_string == "auc":
        return "AUC"
    if snake_case_string == "kullback_leibler_divergence":
        return "KLDivergence"
    # if snake_case_string == "dice_coef":
    #     return "DiceCoefficient"
    if snake_case_string == "unscaled_mae":
        return "UnscaledMAE"
    if snake_case_string == "percent_mae":
        return "PercentMAE"
    if snake_case_string == "logcosh":
        return "LogCoshError"
    return re.sub("_.", lambda x: x.group()[1].upper(), snake_case_string.title())


@contextmanager
def context_cwd(path: Path):
    _cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(_cwd)


def autodetect_encoding(
    path: str, return_encoding: bool = False
) -> Union[str, Tuple[str, str]]:
    available = list(encodings_aliases)
    available.insert(0, "utf8")
    available.insert(1, "windows_1251")

    output = None
    with open(path, "rb") as txt_file_ref:
        content = txt_file_ref.read()
        for encoding in available:
            try:
                output = content.decode(encoding)
                break
            except UnicodeDecodeError:
                pass

    if return_encoding:
        return output, encoding
    else:
        return output


def _get_temppath() -> Path:
    def uuid_path() -> Path:
        return Path(TMP_DIR, str(uuid4()))

    path_dir = uuid_path()
    while path_dir.is_dir() or path_dir.is_file():
        path_dir = uuid_path()

    return path_dir


def get_tempdir(create: bool = True) -> Path:
    path_dir = _get_temppath()
    if create:
        path_dir.mkdir()
    return path_dir


def get_tempfile(create: bool = True) -> Path:
    path_file = _get_temppath()
    if create:
        path_file.touch()
    return path_file


def check_error(error_in: Exception, error_target: str, error_method: str) -> Exception:
    if issubclass(error_in.__class__, TerraBaseException):
        return error_in
    else:
        return NotDescribedException(error_target, error_method).with_traceback(error_in.__traceback__)
