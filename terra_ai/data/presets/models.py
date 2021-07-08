"""
Предустановки моделей
"""

from ..modeling.model import ModelsGroupsList


ModelsGroups = ModelsGroupsList(
    [
        {
            "alias": "preset",
            "name": "Предустановленные",
        },
        {
            "alias": "custom",
            "name": "Собственные",
        },
    ]
)
