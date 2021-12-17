from . import BaseSTT


class Data(BaseSTT.Data):
    class Meta:
        source = BaseSTT.DataList
