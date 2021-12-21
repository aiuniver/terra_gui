import logging

from enum import Enum


class FrontTypeMessage(str, Enum):
    success = "success"
    error = "error"
    warning = "warning"
    info = "info"


logger = logging.getLogger("terra")
