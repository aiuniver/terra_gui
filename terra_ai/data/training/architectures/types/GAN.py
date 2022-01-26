from typing import Optional

from ...checkpoint import CheckpointGANData
from . import Base


class ParametersData(Base.ParametersData):
    checkpoint: Optional[CheckpointGANData]
