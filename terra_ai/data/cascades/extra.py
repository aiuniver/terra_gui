from enum import Enum


class BlockGroupChoice(str, Enum):
    InputData = "InputData"
    OutputData = "OutputData"
    Model = "Model"
    Function = "Function"
    Custom = "Custom"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockGroupChoice))
