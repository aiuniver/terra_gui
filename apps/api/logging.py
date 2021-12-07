import time
import logging

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from pydantic.types import PositiveInt

from django.utils.log import ServerFormatter


class LevelnameColor(str, Enum):
    D = "0;35"
    I = "0;32"
    W = "0;33"
    E = "0;31"
    C = "30;101"


class LevelnameChoice(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogData(BaseModel):
    level: LevelnameChoice
    time: PositiveInt
    title: str
    message: Optional[str]


class TerraConsoleFormatter(ServerFormatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime(self.default_time_format, ct)
        return s

    def format(self, record):
        levelname_short = record.levelname[0]

        if self.uses_levelname_short() and not hasattr(record, "levelname_short"):
            record.levelname_short = levelname_short

        if self.uses_levelname_color() and not hasattr(record, "levelname_color"):
            record.levelname_color = LevelnameColor[levelname_short]

        return super().format(record)

    def uses_levelname_color(self):
        return self._fmt.find("{levelname_color}") >= 0

    def uses_levelname_short(self):
        return self._fmt.find("{levelname_short}") >= 0


class TerraConsoleHandler(logging.Handler):
    def emit(self, record):
        print(self.format(record))
