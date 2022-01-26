from typing import Optional

from ....mixins import BaseMixinData
from ...checkpoint import CheckpointGANData


class ParametersData(BaseMixinData):
    checkpoint: Optional[CheckpointGANData]
