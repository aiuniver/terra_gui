from ....mixins import BaseMixinData
from ..extra import ShowImagesChoice


class ParametersData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_loss_for_classes: bool = True
    plot_metric_for_classes: bool = True
    plot_final: bool = True
    show_images: ShowImagesChoice = ShowImagesChoice.Best
