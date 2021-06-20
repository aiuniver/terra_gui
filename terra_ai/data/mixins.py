import re

from typing import List, Union, Optional
from pydantic import BaseModel, validator


class BaseMixinData(BaseModel):
    def __init__(self, **data):
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if UniqueListMixinData in __type.__mro__:
                data.update({__name: __type(data.get(__name, __type()))})
        super().__init__(**data)

    def dict(self, **kwargs):
        data = super().dict()
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if UniqueListMixinData in __type.__mro__:
                data.update(
                    {
                        __name: list(
                            map(lambda item: item.dict(), data.get(__name, __type()))
                        )
                    }
                )
        return data


class AliasMixinData(BaseMixinData):
    alias: str

    @validator("alias", allow_reuse=True)
    def validate_alias(cls, value):
        if not re.match("^[a-z]+[a-z0-9_]*$", value):
            raise ValueError(
                f'{value}: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
            )
        return value


class UniqueListMixinData(List):
    class Meta:
        source: BaseMixinData = BaseMixinData
        identifier: str

    def __init__(self, data: Optional[List[Union[dict, Meta.source]]] = None):
        if not data:
            data = []
        try:
            data = list(map(lambda item: self.Meta.source(**item), data))
        except TypeError:
            pass
        __data = []
        for item in data:
            if item.dict().get(self.Meta.identifier) not in list(
                map(lambda item: item.dict().get(self.Meta.identifier), __data)
            ):
                __data.append(item)
        super().__init__(__data)

    @property
    def ids(self) -> list:
        if not len(self):
            return []
        if self.Meta.identifier not in self[0].schema().get("properties").keys():
            raise AttributeError(
                f'Identifier "{self.Meta.identifier}" is undefined as attribute of {self.Meta.source}'
            )
        return list(map(lambda item: item.dict().get(self.Meta.identifier), self))

    def get(self, name: str) -> Optional[Meta.source]:
        __ids = self.ids
        if name not in __ids:
            return None
        return self[__ids.index(name)]

    def dict(self) -> List[dict]:
        return list(map(lambda item: item.dict(), self))

    def json(self) -> str:
        return list(map(lambda item: item.json(), self))

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
