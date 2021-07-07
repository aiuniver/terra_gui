import json

from typing import List, Union, Optional, Any
from pydantic import BaseModel

from .types import AliasType
from .exceptions import (
    UniqueListIdentifierException,
    UniqueListUndefinedIdentifierException,
)


class BaseMixinData(BaseModel):
    def __init__(self, **data):
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if hasattr(__type, "__mro__") and UniqueListMixin in __type.__mro__:
                data.update({__name: __type(data.get(__name, __type()))})
        super().__init__(**data)

    def dict(self, **kwargs):
        data = super().dict()
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if hasattr(__type, "__mro__") and UniqueListMixin in __type.__mro__:
                __value = map(lambda item: item.dict(), data.get(__name, __type()))
                data.update({__name: list(__value)})
        return data


class AliasMixinData(BaseMixinData):
    alias: AliasType


class UniqueListMixin(List):
    class Meta:
        source: BaseMixinData = BaseMixinData
        identifier: str

    @property
    def ids(self) -> list:
        if not len(self):
            return []
        if self.Meta.identifier not in self[0].schema().get("properties").keys():
            raise UniqueListIdentifierException(self.Meta.identifier, self.Meta.source)
        return list(map(lambda item: getattr(item, self.Meta.identifier), self))

    def __init__(self, data: Optional[List[Union[dict, Meta.source]]] = None):
        if data is None:
            data = []
        data = list(map(lambda item: self.Meta.source(**item), data))
        __data = []
        for item in data:
            __identifier = getattr(self.Meta, "identifier", None)
            if __identifier is None:
                raise UniqueListUndefinedIdentifierException(self.Meta.source)
            if item.dict().get(__identifier) not in list(
                map(lambda item: item.dict().get(self.Meta.identifier), __data)
            ):
                __data.append(item)
        super().__init__(__data)

    def __iadd__(self, *args, **kwargs):
        if isinstance(args[0], UniqueListMixin):
            for item in args[0]:
                self.append(item)
        return self

    def get(self, name: Any) -> Optional[Meta.source]:
        __ids = self.ids
        if name not in __ids:
            return None
        return self[__ids.index(name)]

    def dict(self) -> List[dict]:
        return list(map(lambda item: item.dict(), self))

    def json(self, **kwargs) -> str:
        __items = []
        for __item in self:
            __items.append(json.loads(__item.json()))
        return json.dumps(__items, **kwargs)

    def append(self, __object: Union[dict, Meta.source]):
        try:
            if not isinstance(__object, (self.Meta.source,)):
                __object = self.Meta.source(**__object)
        except TypeError:
            pass
        if isinstance(__object, dict):
            data = __object
        else:
            data = __object.dict()
        __name = data.get(self.Meta.identifier)
        __o = self.get(__name)
        if __o:
            self[self.ids.index(__name)] = __object
        else:
            super().append(__object)

    def insert(self, __index: int, __object: Union[dict, Meta.source]):
        try:
            if not isinstance(__object, (self.Meta.source,)):
                __object = self.Meta.source(**__object)
        except TypeError:
            pass
        if isinstance(__object, dict):
            data = __object
        else:
            data = __object.dict()
        __name = data.get(self.Meta.identifier)
        __o = self.get(__name)
        if __o:
            __o_index = self.ids.index(__name)
            if self.ids.index(__name) == __index:
                self[__o_index] = __object
            else:
                self.pop(__o_index)
                super().insert(__index, __object)
        else:
            super().insert(__index, __object)
