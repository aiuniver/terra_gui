from ....mixins import BaseMixinData
from ...outputs import OutputsList


class ParametersData(BaseMixinData):
    outputs: OutputsList = OutputsList()

