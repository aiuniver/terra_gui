"""
## Параметры типов слоев
"""

from enum import Enum

from ...mixins import BaseMixinData
from ..extra import LayerTypeChoice
from .extra import ActivationChoice

from . import types


class LayerDefaultData(BaseMixinData):
    pass


class LayerMainDefaultData(LayerDefaultData):
    pass


class LayerExtraDefaultData(LayerDefaultData):
    pass


class LayerConfigDefaultData(LayerDefaultData):
    pass


class LayerMixinData(BaseMixinData):
    main: LayerMainDefaultData = LayerMainDefaultData()
    extra: LayerExtraDefaultData = LayerExtraDefaultData()


class LayerInputData(LayerMixinData):
    main: types.Input.ParametersMainData = types.Input.ParametersMainData()
    extra: types.Input.ParametersExtraData = types.Input.ParametersExtraData()


class LayerConv1DData(LayerMixinData):
    main: types.Conv1D.ParametersMainData = types.Conv1D.ParametersMainData(
        filters=32, kernel_size=5
    )
    extra: types.Conv1D.ParametersExtraData = types.Conv1D.ParametersExtraData()


class LayerConv2DData(LayerMixinData):
    main: types.Conv2D.ParametersMainData = types.Conv2D.ParametersMainData(
        filters=32, kernel_size=(3, 3)
    )
    extra: types.Conv2D.ParametersExtraData = types.Conv2D.ParametersExtraData()
    config: types.Conv2D.LayerConfig = types.Conv2D.LayerConfig()


class LayerConv3DData(LayerMixinData):
    main: types.Conv3D.ParametersMainData = types.Conv3D.ParametersMainData(
        filters=32, kernel_size=(3, 3, 3)
    )
    extra: types.Conv3D.ParametersExtraData = types.Conv3D.ParametersExtraData()
    config: types.Conv3D.LayerConfig = types.Conv3D.LayerConfig()


class LayerConv1DTransposeData(LayerMixinData):
    main: types.Conv1DTranspose.ParametersMainData = (
        types.Conv1DTranspose.ParametersMainData(filters=32, kernel_size=1)
    )
    extra: types.Conv1DTranspose.ParametersExtraData = (
        types.Conv1DTranspose.ParametersExtraData()
    )
    config: types.Conv1DTranspose.LayerConfig = types.Conv1DTranspose.LayerConfig()


class LayerConv2DTransposeData(LayerMixinData):
    main: types.Conv2DTranspose.ParametersMainData = (
        types.Conv2DTranspose.ParametersMainData(filters=32, kernel_size=(1, 1))
    )
    extra: types.Conv2DTranspose.ParametersExtraData = (
        types.Conv2DTranspose.ParametersExtraData()
    )
    config: types.Conv2DTranspose.LayerConfig = types.Conv2DTranspose.LayerConfig()


class LayerConv3DTransposeData(LayerMixinData):
    main: types.Conv3DTranspose.ParametersMainData = (
        types.Conv3DTranspose.ParametersMainData(filters=32, kernel_size=(1, 1, 1))
    )
    extra: types.Conv3DTranspose.ParametersExtraData = (
        types.Conv3DTranspose.ParametersExtraData()
    )
    config: types.Conv3DTranspose.LayerConfig = types.Conv3DTranspose.LayerConfig()


class LayerSeparableConv1DData(LayerMixinData):
    main: types.SeparableConv1D.ParametersMainData = (
        types.SeparableConv1D.ParametersMainData(filters=32, kernel_size=1)
    )
    extra: types.SeparableConv1D.ParametersExtraData = (
        types.SeparableConv1D.ParametersExtraData()
    )
    config: types.SeparableConv1D.LayerConfig = types.SeparableConv1D.LayerConfig()


class LayerSeparableConv2DData(LayerMixinData):
    main: types.SeparableConv2D.ParametersMainData = (
        types.SeparableConv2D.ParametersMainData(filters=32, kernel_size=(1, 1))
    )
    extra: types.SeparableConv2D.ParametersExtraData = (
        types.SeparableConv2D.ParametersExtraData()
    )
    config: types.SeparableConv2D.LayerConfig = types.SeparableConv2D.LayerConfig()


class LayerDepthwiseConv2DData(LayerMixinData):
    main: types.DepthwiseConv2D.ParametersMainData = (
        types.DepthwiseConv2D.ParametersMainData(kernel_size=(1, 1))
    )
    extra: types.DepthwiseConv2D.ParametersExtraData = (
        types.DepthwiseConv2D.ParametersExtraData()
    )
    config: types.DepthwiseConv2D.LayerConfig = types.DepthwiseConv2D.LayerConfig()


class LayerMaxPool1DData(LayerMixinData):
    main: types.MaxPool1D.ParametersMainData = types.MaxPool1D.ParametersMainData()
    extra: types.MaxPool1D.ParametersExtraData = types.MaxPool1D.ParametersExtraData()
    config: types.MaxPool1D.LayerConfig = types.MaxPool1D.LayerConfig()


class LayerMaxPool2DData(LayerMixinData):
    main: types.MaxPool2D.ParametersMainData = types.MaxPool2D.ParametersMainData()
    extra: types.MaxPool2D.ParametersExtraData = types.MaxPool2D.ParametersExtraData()
    config: types.MaxPool2D.LayerConfig = types.MaxPool2D.LayerConfig()


class LayerMaxPool3DData(LayerMixinData):
    main: types.MaxPool3D.ParametersMainData = types.MaxPool3D.ParametersMainData()
    extra: types.MaxPool3D.ParametersExtraData = types.MaxPool3D.ParametersExtraData()
    config: types.MaxPool3D.LayerConfig = types.MaxPool3D.LayerConfig()


class LayerAveragePooling1DData(LayerMixinData):
    main: types.AveragePooling1D.ParametersMainData = (
        types.AveragePooling1D.ParametersMainData()
    )
    extra: types.AveragePooling1D.ParametersExtraData = (
        types.AveragePooling1D.ParametersExtraData()
    )
    config: types.AveragePooling1D.LayerConfig = types.AveragePooling1D.LayerConfig()


class LayerAveragePooling2DData(LayerMixinData):
    main: types.AveragePooling2D.ParametersMainData = (
        types.AveragePooling2D.ParametersMainData()
    )
    extra: types.AveragePooling2D.ParametersExtraData = (
        types.AveragePooling2D.ParametersExtraData()
    )
    config: types.AveragePooling2D.LayerConfig = types.AveragePooling2D.LayerConfig()


class LayerAveragePooling3DData(LayerMixinData):
    main: types.AveragePooling3D.ParametersMainData = (
        types.AveragePooling3D.ParametersMainData()
    )
    extra: types.AveragePooling3D.ParametersExtraData = (
        types.AveragePooling3D.ParametersExtraData()
    )
    config: types.AveragePooling3D.LayerConfig = types.AveragePooling3D.LayerConfig()


class LayerUpSampling1DData(LayerMixinData):
    main: types.UpSampling1D.ParametersMainData = (
        types.UpSampling1D.ParametersMainData()
    )
    extra: types.UpSampling1D.ParametersExtraData = (
        types.UpSampling1D.ParametersExtraData()
    )
    config: types.UpSampling1D.LayerConfig = types.UpSampling1D.LayerConfig()


class LayerUpSampling2DData(LayerMixinData):
    main: types.UpSampling2D.ParametersMainData = (
        types.UpSampling2D.ParametersMainData()
    )
    extra: types.UpSampling2D.ParametersExtraData = (
        types.UpSampling2D.ParametersExtraData()
    )
    config: types.UpSampling2D.LayerConfig = types.UpSampling2D.LayerConfig()


class LayerUpSampling3DData(LayerMixinData):
    main: types.UpSampling3D.ParametersMainData = (
        types.UpSampling3D.ParametersMainData()
    )
    extra: types.UpSampling3D.ParametersExtraData = (
        types.UpSampling3D.ParametersExtraData()
    )
    config: types.UpSampling3D.LayerConfig = types.UpSampling3D.LayerConfig()


class LayerLeakyReLUData(LayerMixinData):
    main: types.LeakyReLU.ParametersMainData = types.LeakyReLU.ParametersMainData()
    extra: types.LeakyReLU.ParametersExtraData = types.LeakyReLU.ParametersExtraData()
    config: types.LeakyReLU.LayerConfig = types.LeakyReLU.LayerConfig()


class LayerDropoutData(LayerMixinData):
    main: types.Dropout.ParametersMainData = types.Dropout.ParametersMainData(rate=0.1)
    extra: types.Dropout.ParametersExtraData = types.Dropout.ParametersExtraData()
    config: types.Dropout.LayerConfig = types.Dropout.LayerConfig()


class LayerDenseData(LayerMixinData):
    main: types.Dense.ParametersMainData = types.Dense.ParametersMainData(units=32)
    extra: types.Dense.ParametersExtraData = types.Dense.ParametersExtraData()
    config: types.Dense.LayerConfig = types.Dense.LayerConfig()


class LayerAddData(LayerMixinData):
    main: types.Add.ParametersMainData = types.Add.ParametersMainData()
    extra: types.Add.ParametersExtraData = types.Add.ParametersExtraData()
    config: types.Add.LayerConfig = types.Add.LayerConfig()


class LayerMultiplyData(LayerMixinData):
    main: types.Multiply.ParametersMainData = types.Multiply.ParametersMainData()
    extra: types.Multiply.ParametersExtraData = types.Multiply.ParametersExtraData()
    config: types.Multiply.LayerConfig = types.Multiply.LayerConfig()


class LayerFlattenData(LayerMixinData):
    main: types.Flatten.ParametersMainData = types.Flatten.ParametersMainData()
    extra: types.Flatten.ParametersExtraData = types.Flatten.ParametersExtraData()
    config: types.Flatten.LayerConfig = types.Flatten.LayerConfig()


class LayerConcatenateData(LayerMixinData):
    main: types.Concatenate.ParametersMainData = types.Concatenate.ParametersMainData()
    extra: types.Concatenate.ParametersExtraData = (
        types.Concatenate.ParametersExtraData()
    )
    config: types.Concatenate.LayerConfig = types.Concatenate.LayerConfig()


class LayerReshapeData(LayerMixinData):
    main: types.Reshape.ParametersMainData = types.Reshape.ParametersMainData()
    extra: types.Reshape.ParametersExtraData = types.Reshape.ParametersExtraData()
    config: types.Reshape.LayerConfig = types.Reshape.LayerConfig()


class LayerSoftmaxData(LayerMixinData):
    main: types.Softmax.ParametersMainData = types.Softmax.ParametersMainData()
    extra: types.Softmax.ParametersExtraData = types.Softmax.ParametersExtraData()
    config: types.Softmax.LayerConfig = types.Softmax.LayerConfig()


class LayerReLUData(LayerMixinData):
    main: types.ReLU.ParametersMainData = types.ReLU.ParametersMainData()
    extra: types.ReLU.ParametersExtraData = types.ReLU.ParametersExtraData()
    config: types.ReLU.LayerConfig = types.ReLU.LayerConfig()


class LayerELUData(LayerMixinData):
    main: types.ELU.ParametersMainData = types.ELU.ParametersMainData()
    extra: types.ELU.ParametersExtraData = types.ELU.ParametersExtraData()
    config: types.ELU.LayerConfig = types.ELU.LayerConfig()


class LayerActivationData(LayerMixinData):
    main: types.Activation.ParametersMainData = types.Activation.ParametersMainData(
        activation=ActivationChoice.relu
    )
    extra: types.Activation.ParametersExtraData = types.Activation.ParametersExtraData()
    config: types.Activation.LayerConfig = types.Activation.LayerConfig()


class LayerPReLUData(LayerMixinData):
    main: types.PReLU.ParametersMainData = types.PReLU.ParametersMainData()
    extra: types.PReLU.ParametersExtraData = types.PReLU.ParametersExtraData()
    config: types.PReLU.LayerConfig = types.PReLU.LayerConfig()


class LayerGlobalMaxPool1DData(LayerMixinData):
    main: types.GlobalMaxPool1D.ParametersMainData = (
        types.GlobalMaxPool1D.ParametersMainData()
    )
    extra: types.GlobalMaxPool1D.ParametersExtraData = (
        types.GlobalMaxPool1D.ParametersExtraData()
    )
    config: types.GlobalMaxPool1D.LayerConfig = types.GlobalMaxPool1D.LayerConfig()


class LayerGlobalMaxPool2DData(LayerMixinData):
    main: types.GlobalMaxPool2D.ParametersMainData = (
        types.GlobalMaxPool2D.ParametersMainData()
    )
    extra: types.GlobalMaxPool2D.ParametersExtraData = (
        types.GlobalMaxPool2D.ParametersExtraData()
    )
    config: types.GlobalMaxPool2D.LayerConfig = types.GlobalMaxPool2D.LayerConfig()


class LayerGlobalMaxPool3DData(LayerMixinData):
    main: types.GlobalMaxPool3D.ParametersMainData = (
        types.GlobalMaxPool3D.ParametersMainData()
    )
    extra: types.GlobalMaxPool3D.ParametersExtraData = (
        types.GlobalMaxPool3D.ParametersExtraData()
    )
    config: types.GlobalMaxPool3D.LayerConfig = types.GlobalMaxPool3D.LayerConfig()


class LayerGlobalAveragePooling1DData(LayerMixinData):
    main: types.GlobalAveragePooling1D.ParametersMainData = (
        types.GlobalAveragePooling1D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling1D.ParametersExtraData = (
        types.GlobalAveragePooling1D.ParametersExtraData()
    )
    config: types.GlobalAveragePooling1D.LayerConfig = (
        types.GlobalAveragePooling1D.LayerConfig()
    )


class LayerGlobalAveragePooling2DData(LayerMixinData):
    main: types.GlobalAveragePooling2D.ParametersMainData = (
        types.GlobalAveragePooling2D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling2D.ParametersExtraData = (
        types.GlobalAveragePooling2D.ParametersExtraData()
    )
    config: types.GlobalAveragePooling2D.LayerConfig = (
        types.GlobalAveragePooling2D.LayerConfig()
    )


class LayerGlobalAveragePooling3DData(LayerMixinData):
    main: types.GlobalAveragePooling3D.ParametersMainData = (
        types.GlobalAveragePooling3D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling3D.ParametersExtraData = (
        types.GlobalAveragePooling3D.ParametersExtraData()
    )
    config: types.GlobalAveragePooling3D.LayerConfig = (
        types.GlobalAveragePooling3D.LayerConfig()
    )


class LayerGRUData(LayerMixinData):
    main: types.GRU.ParametersMainData = types.GRU.ParametersMainData(units=32)
    extra: types.GRU.ParametersExtraData = types.GRU.ParametersExtraData()
    config: types.GRU.LayerConfig = types.GRU.LayerConfig()


class LayerLSTMData(LayerMixinData):
    main: types.LSTM.ParametersMainData = types.LSTM.ParametersMainData(units=32)
    extra: types.LSTM.ParametersExtraData = types.LSTM.ParametersExtraData()
    config: types.LSTM.LayerConfig = types.LSTM.LayerConfig()


class LayerEmbeddingData(LayerMixinData):
    main: types.Embedding.ParametersMainData = types.Embedding.ParametersMainData(
        input_dim=20000, output_dim=64
    )
    extra: types.Embedding.ParametersExtraData = types.Embedding.ParametersExtraData()
    config: types.Embedding.LayerConfig = types.Embedding.LayerConfig()


class LayerRepeatVectorData(LayerMixinData):
    main: types.RepeatVector.ParametersMainData = (
        types.RepeatVector.ParametersMainData()
    )
    extra: types.RepeatVector.ParametersExtraData = (
        types.RepeatVector.ParametersExtraData(n=8)
    )
    config: types.RepeatVector.LayerConfig = types.RepeatVector.LayerConfig()


class LayerBatchNormalizationData(LayerMixinData):
    main: types.BatchNormalization.ParametersMainData = (
        types.BatchNormalization.ParametersMainData()
    )
    extra: types.BatchNormalization.ParametersExtraData = (
        types.BatchNormalization.ParametersExtraData()
    )
    config: types.BatchNormalization.LayerConfig = (
        types.BatchNormalization.LayerConfig()
    )


class ParametersTypeMishData(LayerMixinData):
    main: types.Mish.ParametersMainData = types.Mish.ParametersMainData()
    extra: types.Mish.ParametersExtraData = types.Mish.ParametersExtraData()
    config: types.Mish.LayerConfig = types.Mish.LayerConfig()


class ParametersTypeInstanceNormalizationData(LayerMixinData):
    main: types.InstanceNormalization.ParametersMainData = (
        types.InstanceNormalization.ParametersMainData()
    )
    extra: types.InstanceNormalization.ParametersExtraData = (
        types.InstanceNormalization.ParametersExtraData()
    )
    config: types.InstanceNormalization.LayerConfig = (
        types.InstanceNormalization.LayerConfig()
    )


class ParametersTypeZeroPadding2DData(LayerMixinData):
    main: types.ZeroPadding2D.ParametersMainData = (
        types.ZeroPadding2D.ParametersMainData(padding=((1, 1), (1, 1)))
    )
    extra: types.ZeroPadding2D.ParametersExtraData = (
        types.ZeroPadding2D.ParametersExtraData()
    )
    config: types.ZeroPadding2D.LayerConfig = types.ZeroPadding2D.LayerConfig()


class ParametersTypeCropping2DData(LayerMixinData):
    main: types.Cropping2D.ParametersMainData = types.Cropping2D.ParametersMainData(
        cropping=((1, 1), (1, 1))
    )
    extra: types.Cropping2D.ParametersExtraData = types.Cropping2D.ParametersExtraData()
    config: types.Cropping2D.LayerConfig = types.Cropping2D.LayerConfig()


class ParametersTypeAttentionData(LayerMixinData):
    main: types.Attention.ParametersMainData = types.Attention.ParametersMainData()
    extra: types.Attention.ParametersExtraData = types.Attention.ParametersExtraData()
    config: types.Attention.LayerConfig = types.Attention.LayerConfig()


class ParametersTypeAverageData(LayerMixinData):
    main: types.Average.ParametersMainData = types.Average.ParametersMainData()
    extra: types.Average.ParametersExtraData = types.Average.ParametersExtraData()
    config: types.Average.LayerConfig = types.Average.LayerConfig()


class ParametersTypeThresholdedReLUData(LayerMixinData):
    main: types.ThresholdedReLU.ParametersMainData = (
        types.ThresholdedReLU.ParametersMainData()
    )
    extra: types.ThresholdedReLU.ParametersExtraData = (
        types.ThresholdedReLU.ParametersExtraData()
    )
    config: types.ThresholdedReLU.LayerConfig = types.ThresholdedReLU.LayerConfig()


class ParametersTypeRescalingData(LayerMixinData):
    main: types.Rescaling.ParametersMainData = types.Rescaling.ParametersMainData()
    extra: types.Rescaling.ParametersExtraData = types.Rescaling.ParametersExtraData()
    config: types.Rescaling.LayerConfig = types.Rescaling.LayerConfig()


class ParametersTypeResizingData(LayerMixinData):
    main: types.Resizing.ParametersMainData = types.Resizing.ParametersMainData()
    extra: types.Resizing.ParametersExtraData = types.Resizing.ParametersExtraData()
    config: types.Resizing.LayerConfig = types.Resizing.LayerConfig()


Layer = Enum(
    "Layer",
    dict(map(lambda item: (item.name, f"Layer{item.name}Data"), list(LayerTypeChoice))),
    type=str,
)
"""
Список возможных типов параметров слоя
"""
