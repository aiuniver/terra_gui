from enum import Enum


class BlockGroupChoice(str, Enum):
    InputData = "InputData"
    OutputData = "OutputData"
    Model = "Model"
    Function = "Function"
    Custom = "Custom"
