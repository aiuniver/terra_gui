from ....mixins import BaseMixinData


class ParametersData(BaseMixinData):
    show_every_epoch: bool = True
    plot_loss_metric: bool = True
    plot_metric: bool = True
    plot_autocorrelation: bool = True
    plot_pred_and_true: bool = True
    plot_final: bool = True
