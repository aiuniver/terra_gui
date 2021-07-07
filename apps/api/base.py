from typing import Any
from pydantic import BaseModel

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK


class BaseAPIView(APIView):
    pass


class BaseResponseData(BaseModel):
    success: bool = True
    data: Any
    error: Any


class BaseResponse(Response):
    def __init__(self, success=True, data=None, error=None, *args, **kwargs):
        __response = BaseResponseData(success=success, data=data, error=error)
        kwargs.update({"status": HTTP_200_OK})
        super().__init__(data=__response.dict(), *args, **kwargs)
