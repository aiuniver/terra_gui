from ...mixins import BaseMixinData

from pydantic import PositiveFloat


class YoloParameters(BaseMixinData):
    train_lr_init: PositiveFloat = 1e-4
    train_lr_end: PositiveFloat = 1e-6
    yolo_iou_loss_thresh: PositiveFloat = 0.5
    train_warmup_epochs: PositiveFloat = 2
