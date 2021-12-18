import logging

from typing import Any, Optional, List
from pydantic import BaseModel

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK

from apps.api.logging import logs_catcher, LogData, LevelnameChoice


class BaseAPIView(APIView):
    authentication_classes = ()

    @property
    def terra_exchange(self):
        from terra_ai.agent import agent_exchange

        return agent_exchange

    def dispatch(self, request, *args, **kwargs):
        response = super().dispatch(request, *args, **kwargs)
        warnings = response.data.get("warning", [])
        for log in logs_catcher.record:
            if log.levelno != logging.WARNING:
                continue
            title = None
            message = None
            if isinstance(log.msg, str):
                title = log.msg
            elif isinstance(log.msg, tuple) and len(log.msg) == 2:
                title, message = log.msg
            if title:
                warnings.append(
                    LogData(
                        level=LevelnameChoice.WARNING, title=title, message=message
                    ).dict()
                )
        logs_catcher.clear()
        response.data["warning"] = warnings
        return response


class BaseResponseData(BaseModel):
    success: bool = True
    data: Any
    error: Optional[LogData]
    warning: List[LogData] = []


class BaseResponse(Response):
    def __init__(
        self, data=None, error: LogData = None, warning: LogData = None, *args, **kwargs
    ):
        if warning is None:
            warning = []
        __response = BaseResponseData(
            success=(error is None),
            data=data,
            error=error,
            warning=warning,
        )
        kwargs.update({"status": HTTP_200_OK})
        super().__init__(data=__response.dict(), *args, **kwargs)


class BaseResponseSuccess(BaseResponse):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)


class BaseResponseError(BaseResponse):
    def __init__(self, error: LogData, *args, **kwargs):
        super().__init__(error=error, *args, **kwargs)
