import requests

from typing import Any

from django.conf import settings

from rest_framework.status import HTTP_401_UNAUTHORIZED
from rest_framework.exceptions import APIException

from apps.api.exceptions import AuthAPIException


def request(url: str, data: dict = None) -> Any:
    if data is None:
        data = {}
    response = requests.post(
        f"{settings.TERRA_API_URL}{url}",
        json={"config": settings.USER_PORT, **data},
        cookies={"sessionid": settings.USER_SESSION},
    )
    if response.status_code == HTTP_401_UNAUTHORIZED:
        raise AuthAPIException()
    else:
        if response.ok:
            data = response.json()
            if data.get("success"):
                return data.get("data")
            else:
                raise APIException(data.get("error"))
        else:
            response.raise_for_status()
