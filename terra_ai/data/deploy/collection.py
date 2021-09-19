from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.deploy.extra import TaskTypeChoice


class CollectionData(BaseMixinData):
    type: TaskTypeChoice
