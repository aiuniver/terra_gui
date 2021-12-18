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
        logs = logs_catcher.pool + response.data.get("logs", [])
        response.data.update(
            {
                "logs": list(
                    filter(
                        lambda log: log.get("level")
                        in (LevelnameChoice.INFO, LevelnameChoice.WARNING),
                        logs,
                    )
                )
            }
        )
        return response


class BaseResponseData(BaseModel):
    success: bool = True
    data: Any
    error: Optional[LogData]
    logs: List[LogData] = []


class BaseResponse(Response):
    def __init__(
        self, data=None, error: LogData = None, logs: LogData = None, *args, **kwargs
    ):
        if logs is None:
            logs = []
        __response = BaseResponseData(
            success=(error is None),
            data=data,
            error=error,
            logs=logs,
        )
        kwargs.update({"status": HTTP_200_OK})
        super().__init__(data=__response.dict(), *args, **kwargs)


class BaseResponseSuccess(BaseResponse):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)


class BaseResponseError(BaseResponse):
    def __init__(self, error: LogData, *args, **kwargs):
        super().__init__(error=error, *args, **kwargs)
