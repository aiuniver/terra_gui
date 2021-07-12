from ....mixins import BaseMixinData


class ParametersData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_loss_for_classes: bool = True
    plot_metric_for_classes: bool = True
    show_best_images: bool = True
    show_worst_images: bool = False
    plot_final: bool = True
