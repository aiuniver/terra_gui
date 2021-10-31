from pydantic import BaseModel

from ..base import Field
from . import optimizers, architectures


class CallbacksClassificationData(BaseModel):
    show_every_epoch: Field
    plot_loss_metric: Field
    plot_metric: Field
    plot_loss_for_classes: Field
    plot_metric_for_classes: Field
    plot_final: Field
    show_images: Field


class CallbacksSegmentationData(BaseModel):
    show_every_epoch: Field
    plot_loss_metric: Field
    plot_metric: Field
    plot_loss_for_classes: Field
    plot_metric_for_classes: Field
    plot_final: Field
    show_images: Field


class CallbacksRegressionData(BaseModel):
    show_every_epoch: Field
    plot_loss_metric: Field
    plot_metric: Field
    plot_scatter: Field
    plot_final: Field


class CallbacksTimeseriesData(BaseModel):
    show_every_epoch: Field
    plot_loss_metric: Field
    plot_metric: Field
    plot_autocorrelation: Field
    plot_pred_and_true: Field
    plot_final: Field


class CallbacksData(BaseModel):
    Classification: CallbacksClassificationData
    Segmentation: CallbacksSegmentationData
    Regression: CallbacksRegressionData
    Timeseries: CallbacksTimeseriesData


class OptimizersData(BaseModel):
    SGD: optimizers.OptimizerSGDData
    RMSprop: optimizers.OptimizerRMSpropData
    Adam: optimizers.OptimizerAdamData
    Adadelta: optimizers.OptimizerAdadeltaData
    Adagrad: optimizers.OptimizerAdagradData
    Adamax: optimizers.OptimizerAdamaxData
    Nadam: optimizers.OptimizerNadamData
    Ftrl: optimizers.OptimizerFtrlData


class ArchitecturesData(BaseModel):
    Basic: architectures.Basic.GroupsData
    YoloV3: architectures.YoloV3.GroupsData
    YoloV4: architectures.YoloV4.GroupsData


class FormMain(BaseModel):
    architecture: Field
    optimizer: Field
    batch: Field
    epochs: Field
    learning_rate: Field


class Form(BaseModel):
    losses: dict
    metrics: dict
    callbacks: CallbacksData
    optimizers: OptimizersData
    architectures: ArchitecturesData
    form: FormMain
