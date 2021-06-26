"""
## Структура данных тегов
"""

from ..mixins import AliasMixinData, UniqueListMixin


class TagData(AliasMixinData):
    """
    Информация о теге
    """

    name: str
    "Название"


class TagsListData(UniqueListMixin):
    """
    Список тегов, основанных на `TagData`
    ```
    class Meta:
        source = TagData
        identifier = "alias"
    ```
    """

    class Meta:
        source = TagData
        identifier = "alias"
