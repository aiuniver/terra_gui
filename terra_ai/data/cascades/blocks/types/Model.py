from pydantic import DirectoryPath

from terra_ai.data.mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    path: DirectoryPath
    postprocess: bool = True
