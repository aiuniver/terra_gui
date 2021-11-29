import re
import os

from pathlib import Path
from contextlib import contextmanager
from typing import Union, Tuple
from encodings.aliases import aliases as encodings_aliases


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
