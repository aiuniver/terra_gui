from typing import Any

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.deploy.extra import CollectionTypeChoice


class BaseCollection(BaseMixinData):
    type: CollectionTypeChoice
    data: Any


# class
