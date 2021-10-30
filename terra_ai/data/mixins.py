"""
## Расширение структур данных
"""

import json

from typing import List, Union, Optional, Any
from pydantic import BaseModel

from .types import AliasType, IDType
from .exceptions import (
    UniqueListIdentifierException,
    UniqueListUndefinedIdentifierException,
)
from ..exceptions.decorators import error_handler


class BaseMixinData(BaseModel):
    """
    Базовая модель, которая должна применяться ко всем структурам.
    """

    @error_handler
    def __init__(self, **data):
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if hasattr(__type, "__mro__") and UniqueListMixin in __type.__mro__:
                data.update({__name: __type(data.get(__name, __type()))})
        super().__init__(**data)

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        for __name, __field in self.__fields__.items():
            __type = __field.type_
            if hasattr(__type, "__mro__") and UniqueListMixin in __type.__mro__:
                __value = map(lambda item: item.dict(), data.get(__name, __type()))
                data.update({__name: list(__value)})
        return data

    def native(self) -> dict:
        return json.loads(self.json())


class IDMixinData(BaseMixinData):
    """
    Расширение модели идентификатором `id`.
    """

    id: IDType


class AliasMixinData(BaseMixinData):
    """
    Расширение модели идентификатором `alias`.
    """

    alias: AliasType


class UniqueListMixin(List):
    """
    Уникальный список, состоящий из `BaseModel`, идентификатор которого определяется в `Meta`
    ```
    class SomeData(AliasMixinData):
        name: str

    class SomeUniqueList(UniqueListMixin):
        class Meta:
            source = SomeData
            identifier = "alias"

    some_unique_list = SomeUniqueList(
        [
            {
                "alias": "some_alias",
                "name": "Some name",
            },
            {
                "alias": "some_second_alias",
                "name": "Some second name",
            }
        ]
    )

    >> some_unique_list
    [SomeData(alias='some_alias', name='Some name'), SomeData(alias='some_second_alias', name='Some second name')]

    >> type(some_unique_list)
    <class '__main__.SomeUniqueList'>
    ```
    """

    class Meta:
        """
        Мета-данные необходимые для определения типа данных в списке и поля-идентификатора
        ```
        source: BaseModel = BaseModel
        ```
        - может быть любой структурой, основанной на `BaseModel`
        ```
        identifier: str
        ```
        - поле-идентификатор
        """

        source: BaseMixinData = BaseMixinData
        identifier: str

    @property
    def ids(self) -> list:
        """
        Список всех идентификаторов текущего списка элементов
        ```
        >> ids = some_unique_list.ids

        >> ids
        ['some_alias', 'some_second_alias']

        >> type(ids)
        <class 'list'>
        ```
        """
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
        """
        Получение элемента по уникальному идентификатору
        ```
        >> some_element = some_unique_list.get("some_alias")

        >> some_element
        SomeData(alias='some_alias', name='Some name')
        ```
        """
        __ids = self.ids
        if name not in __ids:
            return None
        return self[__ids.index(name)]

    def dict(self) -> List[dict]:
        """
        Получить структуру в виде `List[dict]`
        ```
        >> data_dict = some_unique_list.dict()

        >> data_dict
        [{'alias': 'some_alias', 'name': 'Some name'}, {'alias': 'some_second_alias', 'name': 'Some second name'}]

        >> type(data_dict)
        <class 'list'>
        ```
        """
        return list(map(lambda item: item.dict(), self))

    def json(self, **kwargs) -> str:
        """
        Получить `json`-строку
        ```
        >> data_json = some_unique_list.json()

        >> data_json
        '[{"alias": "some_alias", "name": "Some name"}, {"alias": "some_second_alias", "name": "Some second name"}]'

        >> type(data_json)
        <class 'str'>

        >> data_json_format = some_unique_list.json(indent=2)

        >> print(data_json_format)
        [
          {
            "alias": "some_alias",
            "name": "Some name"
          },
          {
            "alias": "some_second_alias",
            "name": "Some second name"
          }
        ]

        >> type(data_json_format)
        <class 'str'>
        ```
        """
        __items = []
        for __item in self:
            __items.append(json.loads(__item.json()))
        return json.dumps(__items, **kwargs)

    def native(self) -> dict:
        return json.loads(self.json())

    def append(self, __object: Union[dict, Meta.source]):
        """
        Добавить объект в конец списка в случае, если уникальный идентификатор не найден в списке. Если же идентификатор уже существует, то заменяются текущие данные
        ```
        >> some_unique_list.append({"alias": "some_alias_append", "name": "Some name append"})

        >> some_unique_list
        [SomeData(alias='some_alias', name='Some name'), SomeData(alias='some_second_alias', name='Some second name'), SomeData(alias='some_alias_append', name='Some name append')]

        >> some_unique_list.append({"alias": "some_alias", "name": "Some name 2"})

        >> some_unique_list
        [SomeData(alias='some_alias', name='Some name 2'), SomeData(alias='some_second_alias', name='Some second name'), SomeData(alias='some_alias_append', name='Some name append')]
        ```
        """
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
        """
        Добавить объект в конкретную позицию списка. Если уникальный идентификатор уже существует, то старое значение удаляется, а новое добавляется на конкретную позицию
        ```
        >> some_unique_list.insert(1, {"alias": "some_alias_insert", "name": "Some name insert"})

        >> some_unique_list
        [SomeData(alias='some_alias', name='Some name'), SomeData(alias='some_alias_insert', name='Some name insert'), SomeData(alias='some_second_alias', name='Some second name')]

        >> some_unique_list.insert(1, {"alias": "some_alias", "name": "Some name 2"})

        >> some_unique_list
        [SomeData(alias='some_alias_insert', name='Some name insert'), SomeData(alias='some_alias', name='Some name 2'), SomeData(alias='some_second_alias', name='Some second name')]
        ```
        """
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
