from . import YoloV3


class Data(YoloV3.Data):
    class Meta:
        source = YoloV3.DataList
