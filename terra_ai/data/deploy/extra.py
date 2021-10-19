from enum import Enum
from . import tasks


class EnvVersionChoice(str, Enum):
    v1 = "v1"


class DeployTypeChoice(str, Enum):
    ImageSegmentation = "ImageSegmentation"
    ImageClassification = "ImageClassification"
    TextSegmentation = "TextSegmentation"
    TextClassification = "TextClassification"

    @property
    def dataclass(self) -> tasks.DeployBase:
        return getattr(tasks, f"Deploy{self.value}")
