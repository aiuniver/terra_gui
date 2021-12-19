import time
import json
import logging

from enum import Enum
from time import mktime
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from pydantic.types import PositiveInt

from django.utils.log import ServerFormatter

from terra_ai.logging import FrontTypeMessage


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
    time: Optional[PositiveInt]
    title: str
    message: Optional[str]
    type: Optional[FrontTypeMessage]

    def __init__(self, **data):
        data["time"] = mktime(datetime.now().timetuple())
        super().__init__(**data)


class TerraLogsCatcher:
    _logs: List[dict] = []
    _pool: List[dict] = []

    @property
    def logs(self) -> List[dict]:
        return self._logs

    @property
    def pool(self) -> List[dict]:
        _logs = self._pool
        self._pool = []
        self._logs = _logs + self._logs
        return _logs

    def push(self, log: LogData):
        self._pool.insert(0, json.loads(log.json(ensure_ascii=False)))


logs_catcher = TerraLogsCatcher()


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


class TerraLogsCatcherHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        logs_catcher.push(
            LogData(
                level=record.levelname,
                title=record.getMessage(),
                message=str(record.exc_info[1]) if record.exc_info else None,
                type=getattr(record, "type", None),
            )
        )
