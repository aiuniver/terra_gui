from typing import Dict, List
from pydantic import BaseModel

from .extra import LayerInputTypeChoice, LayerOutputTypeChoice


class DatasetLayerParameters(BaseModel):
    pass


class DatasetLayersParameters(BaseModel):
    input: Dict[LayerInputTypeChoice, List]
    output: Dict[LayerOutputTypeChoice, List]
