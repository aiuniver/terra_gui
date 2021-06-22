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
    Список тегов, основанных на [`data.dts.tags.TagData`](#data.dts.tags.TagData)
    ```
    class Meta:
        source = data.dts.tags.TagData
        identifier = "alias"
    ```
    """

    class Meta:
        source = TagData
        identifier = "alias"
