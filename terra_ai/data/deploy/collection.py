from typing import List

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.deploy.tasks import BaseCollection
from terra_ai.data.deploy.extra import TaskTypeChoice


class CollectionData(BaseMixinData):
    type: TaskTypeChoice
    data: List[BaseCollection] = []
