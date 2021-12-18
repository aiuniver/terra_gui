from . import BaseTTS


class Data(BaseTTS.Data):
    class Meta:
        source = BaseTTS.DataList
