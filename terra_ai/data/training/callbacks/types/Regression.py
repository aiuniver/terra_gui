from ....mixins import BaseMixinData


class ParametersData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_scatter: bool = True
    plot_final: bool = True
