"""
Предустановки моделей
"""

from ..modeling.model import ModelsGroupsList


groups = ModelsGroupsList(
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
