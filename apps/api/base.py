from typing import Any
from pydantic import BaseModel, ValidationError
from dict_recursive_update import recursive_update
from collections.abc import MutableMapping

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK




def flatten_dict(d: MutableMapping, parent_key: str = '[', sep: str = '][') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + k + sep if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))

    return dict(items)


class BaseAPIView(APIView):
    pass


class BaseResponseData(BaseModel):
    success: bool = True
    data: Any
    error: Any


class BaseResponse(Response):
    def __init__(self, data=None, error=None, *args, **kwargs):
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
    def __init__(self, error=None, *args, **kwargs):
        super().__init__(error=error, *args, **kwargs)


class BaseResponseErrorGeneral(BaseResponseError):
    def __init__(self, error=None, *args, **kwargs):
        if isinstance(error, dict):
            error = list(filter(None, [str(error.get("detail", ""))]))
        super().__init__(error={"general": error}, *args, **kwargs)


class BaseResponseErrorFields(BaseResponseError):
    def __init__(self, error=None, *args, **kwargs):
        if isinstance(error, ValidationError):
            __errors = {}
            for __error in error.errors():
                __locs = __error.get("loc", ())
                __current_errors = __errors.get(__locs[0], {})
                __locs = __locs[1:]
                while __locs:
                    __loc = __locs[0]
                    __current_errors = __current_errors.get(__loc, {})
                    __locs = __locs[1:]
                if not __current_errors:
                    __current_errors = []
                __loc_dict = __current_errors + [__error.get("msg")]
                __locs = __error.get("loc", ())
                while __locs:
                    __loc = __locs[-1]
                    __loc_dict = {__loc: __loc_dict}
                    __locs = __locs[:-1]
                __errors = recursive_update(__errors, __loc_dict)
            error = __errors
        buff = flatten_dict({"fields": error})
        error = dict((key[:len(key)-1], value) for (key, value) in buff.items())

        super().__init__(error=error, *args, **kwargs)
