from typing import Optional, Any

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.dataset import DatasetData


class ParametersMainData(BaseMixinData):
    dataset: Any
