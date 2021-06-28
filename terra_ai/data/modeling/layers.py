"""
## Структура данных слоев
"""

from ..mixins import AliasMixinData, UniqueListMixin


class LayerData(AliasMixinData):
    """
    Данные слоя
    """

    name: str
    "Название"


class LayersList(UniqueListMixin):
    """
    Список слоев, основанных на `LayerData`
    ```
    class Meta:
        source = LayerData
        identifier = "alias"
    ```
    """

    class Meta:
        source = LayerData
        identifier = "alias"
