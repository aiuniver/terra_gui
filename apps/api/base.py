from typing import Any, Optional
from pydantic import BaseModel

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK

from apps.api.logging import LogData


class BaseAPIView(APIView):
    authentication_classes = ()


class BaseResponseData(BaseModel):
    success: bool = True
    data: Any
    error: Optional[LogData]


class BaseResponse(Response):
    def __init__(self, data=None, error: LogData = None, *args, **kwargs):
        __response = BaseResponseData(
            success=(error is None),
            data=data,
            error=error,
        )
        kwargs.update({"status": HTTP_200_OK})
        super().__init__(data=__response.dict(), *args, **kwargs)


class BaseResponseSuccess(BaseResponse):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data=data, *args, **kwargs)


class BaseResponseError(BaseResponse):
    def __init__(self, error: LogData, *args, **kwargs):
        super().__init__(error=error, *args, **kwargs)
