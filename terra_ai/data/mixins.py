import re

from typing import List, Union, Any, Optional
from pydantic import BaseModel, validator


class AliasMixinData(BaseModel):
    alias: str

    @validator("alias", allow_reuse=True)
    def validate_alias(cls, value):
        if not re.match("^[a-z]+[a-z0-9_]*$", value):
            raise ValueError(
                f'{value}: It is allowed to use only lowercase latin characters, numbers and the "_" sign, must always begin with a latin character'
            )
        return value


class ListOfDictMixinData(BaseModel):
    class Meta:
        lists_of_dict: Optional[list] = []

    def __init__(self, **data):
        if self.Meta.lists_of_dict:
            for name in self.Meta.lists_of_dict:
                model_type = self.__fields__.get(name).type_
                data.update({name: model_type(data.get(name, model_type()))})
        super().__init__(**data)

    def dict(self, **kwargs):
        data = super().dict()
        if self.Meta.lists_of_dict:
            for name in self.Meta.lists_of_dict:
                model_type = self.__fields__.get(name).type_
                data.update(
                    {
                        name: list(
                            map(lambda item: item.dict(), data.get(name, model_type()))
                        )
                    }
                )
        return data


class ListMixinData(List):
    class Meta:
        source: Any = dict
        identifier: Optional[str]

    def __init__(self, data: List[Union[dict, Meta.source]] = None):
        if not data:
            data = []
        try:
            data = list(map(lambda item: self.Meta.source(**item), data))
        except TypeError:
            pass
        super().__init__(data)

    @property
    def ids(self) -> list:
        if not len(self):
            return []
        if self.Meta.source == dict:
            data = self
            if self.Meta.identifier not in self[0].keys():
                raise AttributeError(
                    f'Identifier "{self.Meta.identifier}" is undefined as key of dict'
                )
        else:
            data = list(map(lambda item: item.dict(), self))
            if self.Meta.identifier not in self[0].schema().get("properties").keys():
                raise AttributeError(
                    f'Identifier "{self.Meta.identifier}" is undefined as attribute of {self.Meta.source}'
                )
        return list(map(lambda item: item.get(self.Meta.identifier), data))

    def get(self, name: str) -> Optional[Union[dict, Meta.source]]:
        __ids = self.ids
        if name not in __ids:
            return None
        return self[__ids.index(name)]

    def dict(self) -> List[dict]:
        return list(map(lambda item: item.dict(), self))

    def json(self) -> List[dict]:
        return list(map(lambda item: item.json(), self))

    def append(self, __object: Union[dict, Meta.source]):
        try:
            if not isinstance(__object, (self.Meta.source,)):
                __object = self.Meta.source(**__object)
        except TypeError:
            pass
        super().append(__object)

    def insert(self, __index: int, __object: Union[dict, Meta.source]):
        try:
            if not isinstance(__object, (self.Meta.source,)):
                __object = self.Meta.source(**__object)
        except TypeError:
            pass
        super().insert(__index, __object)
