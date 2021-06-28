"""
Предустановки моделей
"""

from ..modeling.models import ModelsGroupsList


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
