from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel


class ShowImagesChoice(str, Enum):
    Best = "Лучшие"
    Worst = "Худшие"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ShowImagesChoice))

    @staticmethod
    def names() -> list:
        return list(map(lambda item: item.name, ShowImagesChoice))

    @staticmethod
    def options() -> List[Tuple[str, str]]:
        return list(map(lambda item: (item.name, item.value), ShowImagesChoice))


class GroupData(BaseModel):
    label: str
    collapsable: bool
    collapsed: bool
