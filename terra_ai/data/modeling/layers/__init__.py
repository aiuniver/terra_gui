"""
## Параметры типов слоев
"""
import os
import shutil
from enum import Enum
from typing import Any, Optional
from pathlib import Path
from pydantic import validator
from pydantic.types import FilePath

from terra_ai import settings
from terra_ai.progress import utils as progress_utils
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.modeling.layers import types
from terra_ai.data.modeling.layers.extra import (
    ActivationChoice,
    LayerConfigData,
    YOLOModeChoice,
)

WEIGHT_FILES = {
    YOLOModeChoice.YOLOv3: "yolov3.weights",
    YOLOModeChoice.YOLOv4: "yolov4.weights",
}


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

    @property
    def config(self) -> LayerConfigData:
        return getattr(types, Layer(self.__class__.__name__).name).LayerConfig

    @property
    def defaults(self) -> Any:
        return self.__class__()

    @property
    def merged(self) -> dict:
        data = self.main.native()
        data.update(**self.extra.native())
        return data


class LayerInputData(LayerMixinData):
    main: types.Input.ParametersMainData = types.Input.ParametersMainData()
    extra: types.Input.ParametersExtraData = types.Input.ParametersExtraData()


class LayerConv1DData(LayerMixinData):
    main: types.Conv1D.ParametersMainData = types.Conv1D.ParametersMainData()
    extra: types.Conv1D.ParametersExtraData = types.Conv1D.ParametersExtraData()


class LayerConv2DData(LayerMixinData):
    main: types.Conv2D.ParametersMainData = types.Conv2D.ParametersMainData()
    extra: types.Conv2D.ParametersExtraData = types.Conv2D.ParametersExtraData()


class LayerConv3DData(LayerMixinData):
    main: types.Conv3D.ParametersMainData = types.Conv3D.ParametersMainData()
    extra: types.Conv3D.ParametersExtraData = types.Conv3D.ParametersExtraData()


class LayerConv1DTransposeData(LayerMixinData):
    main: types.Conv1DTranspose.ParametersMainData = (
        types.Conv1DTranspose.ParametersMainData()
    )

    extra: types.Conv1DTranspose.ParametersExtraData = (
        types.Conv1DTranspose.ParametersExtraData()
    )


class LayerConv2DTransposeData(LayerMixinData):
    main: types.Conv2DTranspose.ParametersMainData = (
        types.Conv2DTranspose.ParametersMainData()
    )
    extra: types.Conv2DTranspose.ParametersExtraData = (
        types.Conv2DTranspose.ParametersExtraData()
    )


class LayerConv3DTransposeData(LayerMixinData):
    main: types.Conv3DTranspose.ParametersMainData = (
        types.Conv3DTranspose.ParametersMainData()
    )
    extra: types.Conv3DTranspose.ParametersExtraData = (
        types.Conv3DTranspose.ParametersExtraData()
    )


class LayerConvLSTM1DData(LayerMixinData):
    main: types.ConvLSTM1D.ParametersMainData = types.ConvLSTM1D.ParametersMainData()
    extra: types.ConvLSTM1D.ParametersExtraData = types.ConvLSTM1D.ParametersExtraData()


class LayerConvLSTM2DData(LayerMixinData):
    main: types.ConvLSTM2D.ParametersMainData = types.ConvLSTM2D.ParametersMainData()
    extra: types.ConvLSTM2D.ParametersExtraData = types.ConvLSTM2D.ParametersExtraData()


class LayerConvLSTM3DData(LayerMixinData):
    main: types.ConvLSTM3D.ParametersMainData = types.ConvLSTM3D.ParametersMainData()
    extra: types.ConvLSTM3D.ParametersExtraData = types.ConvLSTM3D.ParametersExtraData()


class LayerSeparableConv1DData(LayerMixinData):
    main: types.SeparableConv1D.ParametersMainData = (
        types.SeparableConv1D.ParametersMainData()
    )
    extra: types.SeparableConv1D.ParametersExtraData = (
        types.SeparableConv1D.ParametersExtraData()
    )


class LayerSeparableConv2DData(LayerMixinData):
    main: types.SeparableConv2D.ParametersMainData = (
        types.SeparableConv2D.ParametersMainData()
    )
    extra: types.SeparableConv2D.ParametersExtraData = (
        types.SeparableConv2D.ParametersExtraData()
    )


class LayerDepthwiseConv2DData(LayerMixinData):
    main: types.DepthwiseConv2D.ParametersMainData = (
        types.DepthwiseConv2D.ParametersMainData()
    )
    extra: types.DepthwiseConv2D.ParametersExtraData = (
        types.DepthwiseConv2D.ParametersExtraData()
    )


class LayerMaxPool1DData(LayerMixinData):
    main: types.MaxPool1D.ParametersMainData = types.MaxPool1D.ParametersMainData()
    extra: types.MaxPool1D.ParametersExtraData = types.MaxPool1D.ParametersExtraData()


class LayerMaxPool2DData(LayerMixinData):
    main: types.MaxPool2D.ParametersMainData = types.MaxPool2D.ParametersMainData()
    extra: types.MaxPool2D.ParametersExtraData = types.MaxPool2D.ParametersExtraData()


class LayerMaxPool3DData(LayerMixinData):
    main: types.MaxPool3D.ParametersMainData = types.MaxPool3D.ParametersMainData()
    extra: types.MaxPool3D.ParametersExtraData = types.MaxPool3D.ParametersExtraData()


class LayerAveragePooling1DData(LayerMixinData):
    main: types.AveragePooling1D.ParametersMainData = (
        types.AveragePooling1D.ParametersMainData()
    )
    extra: types.AveragePooling1D.ParametersExtraData = (
        types.AveragePooling1D.ParametersExtraData()
    )


class LayerAveragePooling2DData(LayerMixinData):
    main: types.AveragePooling2D.ParametersMainData = (
        types.AveragePooling2D.ParametersMainData()
    )
    extra: types.AveragePooling2D.ParametersExtraData = (
        types.AveragePooling2D.ParametersExtraData()
    )


class LayerAveragePooling3DData(LayerMixinData):
    main: types.AveragePooling3D.ParametersMainData = (
        types.AveragePooling3D.ParametersMainData()
    )
    extra: types.AveragePooling3D.ParametersExtraData = (
        types.AveragePooling3D.ParametersExtraData()
    )


class LayerUpSampling1DData(LayerMixinData):
    main: types.UpSampling1D.ParametersMainData = (
        types.UpSampling1D.ParametersMainData()
    )
    extra: types.UpSampling1D.ParametersExtraData = (
        types.UpSampling1D.ParametersExtraData()
    )


class LayerUpSampling2DData(LayerMixinData):
    main: types.UpSampling2D.ParametersMainData = (
        types.UpSampling2D.ParametersMainData()
    )
    extra: types.UpSampling2D.ParametersExtraData = (
        types.UpSampling2D.ParametersExtraData()
    )


class LayerUpSampling3DData(LayerMixinData):
    main: types.UpSampling3D.ParametersMainData = (
        types.UpSampling3D.ParametersMainData()
    )
    extra: types.UpSampling3D.ParametersExtraData = (
        types.UpSampling3D.ParametersExtraData()
    )


class LayerLeakyReLUData(LayerMixinData):
    main: types.LeakyReLU.ParametersMainData = types.LeakyReLU.ParametersMainData()
    extra: types.LeakyReLU.ParametersExtraData = types.LeakyReLU.ParametersExtraData()


class LayerDropoutData(LayerMixinData):
    main: types.Dropout.ParametersMainData = types.Dropout.ParametersMainData()
    extra: types.Dropout.ParametersExtraData = types.Dropout.ParametersExtraData()


class LayerDenseData(LayerMixinData):
    main: types.Dense.ParametersMainData = types.Dense.ParametersMainData()
    extra: types.Dense.ParametersExtraData = types.Dense.ParametersExtraData()


class LayerAddData(LayerMixinData):
    main: types.Add.ParametersMainData = types.Add.ParametersMainData()
    extra: types.Add.ParametersExtraData = types.Add.ParametersExtraData()


class LayerMultiplyData(LayerMixinData):
    main: types.Multiply.ParametersMainData = types.Multiply.ParametersMainData()
    extra: types.Multiply.ParametersExtraData = types.Multiply.ParametersExtraData()


class LayerFlattenData(LayerMixinData):
    main: types.Flatten.ParametersMainData = types.Flatten.ParametersMainData()
    extra: types.Flatten.ParametersExtraData = types.Flatten.ParametersExtraData()


class LayerConcatenateData(LayerMixinData):
    main: types.Concatenate.ParametersMainData = types.Concatenate.ParametersMainData()
    extra: types.Concatenate.ParametersExtraData = (
        types.Concatenate.ParametersExtraData()
    )


class LayerReshapeData(LayerMixinData):
    main: types.Reshape.ParametersMainData = types.Reshape.ParametersMainData()
    extra: types.Reshape.ParametersExtraData = types.Reshape.ParametersExtraData()


class LayerSoftmaxData(LayerMixinData):
    main: types.Softmax.ParametersMainData = types.Softmax.ParametersMainData()
    extra: types.Softmax.ParametersExtraData = types.Softmax.ParametersExtraData()


class LayerReLUData(LayerMixinData):
    main: types.ReLU.ParametersMainData = types.ReLU.ParametersMainData()
    extra: types.ReLU.ParametersExtraData = types.ReLU.ParametersExtraData()


class LayerELUData(LayerMixinData):
    main: types.ELU.ParametersMainData = types.ELU.ParametersMainData()
    extra: types.ELU.ParametersExtraData = types.ELU.ParametersExtraData()


class LayerActivationData(LayerMixinData):
    main: types.Activation.ParametersMainData = types.Activation.ParametersMainData(
        activation=ActivationChoice.relu
    )
    extra: types.Activation.ParametersExtraData = types.Activation.ParametersExtraData()


class LayerPReLUData(LayerMixinData):
    main: types.PReLU.ParametersMainData = types.PReLU.ParametersMainData()
    extra: types.PReLU.ParametersExtraData = types.PReLU.ParametersExtraData()


class LayerGlobalMaxPool1DData(LayerMixinData):
    main: types.GlobalMaxPool1D.ParametersMainData = (
        types.GlobalMaxPool1D.ParametersMainData()
    )
    extra: types.GlobalMaxPool1D.ParametersExtraData = (
        types.GlobalMaxPool1D.ParametersExtraData()
    )


class LayerGlobalMaxPool2DData(LayerMixinData):
    main: types.GlobalMaxPool2D.ParametersMainData = (
        types.GlobalMaxPool2D.ParametersMainData()
    )
    extra: types.GlobalMaxPool2D.ParametersExtraData = (
        types.GlobalMaxPool2D.ParametersExtraData()
    )


class LayerGlobalMaxPool3DData(LayerMixinData):
    main: types.GlobalMaxPool3D.ParametersMainData = (
        types.GlobalMaxPool3D.ParametersMainData()
    )
    extra: types.GlobalMaxPool3D.ParametersExtraData = (
        types.GlobalMaxPool3D.ParametersExtraData()
    )


class LayerGlobalAveragePooling1DData(LayerMixinData):
    main: types.GlobalAveragePooling1D.ParametersMainData = (
        types.GlobalAveragePooling1D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling1D.ParametersExtraData = (
        types.GlobalAveragePooling1D.ParametersExtraData()
    )


class LayerGlobalAveragePooling2DData(LayerMixinData):
    main: types.GlobalAveragePooling2D.ParametersMainData = (
        types.GlobalAveragePooling2D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling2D.ParametersExtraData = (
        types.GlobalAveragePooling2D.ParametersExtraData()
    )


class LayerGlobalAveragePooling3DData(LayerMixinData):
    main: types.GlobalAveragePooling3D.ParametersMainData = (
        types.GlobalAveragePooling3D.ParametersMainData()
    )
    extra: types.GlobalAveragePooling3D.ParametersExtraData = (
        types.GlobalAveragePooling3D.ParametersExtraData()
    )


class LayerGRUData(LayerMixinData):
    main: types.GRU.ParametersMainData = types.GRU.ParametersMainData()
    extra: types.GRU.ParametersExtraData = types.GRU.ParametersExtraData()


class LayerLSTMData(LayerMixinData):
    main: types.LSTM.ParametersMainData = types.LSTM.ParametersMainData()
    extra: types.LSTM.ParametersExtraData = types.LSTM.ParametersExtraData()


class LayerEmbeddingData(LayerMixinData):
    main: types.Embedding.ParametersMainData = types.Embedding.ParametersMainData()
    extra: types.Embedding.ParametersExtraData = types.Embedding.ParametersExtraData()


class LayerRepeatVectorData(LayerMixinData):
    main: types.RepeatVector.ParametersMainData = (
        types.RepeatVector.ParametersMainData()
    )
    extra: types.RepeatVector.ParametersExtraData = (
        types.RepeatVector.ParametersExtraData()
    )


class LayerNormalizationData(LayerMixinData):
    main: types.Normalization.ParametersMainData = (
        types.Normalization.ParametersMainData()
    )
    extra: types.Normalization.ParametersExtraData = (
        types.Normalization.ParametersExtraData()
    )


class LayerBatchNormalizationData(LayerMixinData):
    main: types.BatchNormalization.ParametersMainData = (
        types.BatchNormalization.ParametersMainData()
    )
    extra: types.BatchNormalization.ParametersExtraData = (
        types.BatchNormalization.ParametersExtraData()
    )


class LayerMishData(LayerMixinData):
    main: types.Mish.ParametersMainData = types.Mish.ParametersMainData()
    extra: types.Mish.ParametersExtraData = types.Mish.ParametersExtraData()


class LayerInstanceNormalizationData(LayerMixinData):
    main: types.InstanceNormalization.ParametersMainData = (
        types.InstanceNormalization.ParametersMainData()
    )
    extra: types.InstanceNormalization.ParametersExtraData = (
        types.InstanceNormalization.ParametersExtraData()
    )


class LayerZeroPadding2DData(LayerMixinData):
    main: types.ZeroPadding2D.ParametersMainData = (
        types.ZeroPadding2D.ParametersMainData()
    )
    extra: types.ZeroPadding2D.ParametersExtraData = (
        types.ZeroPadding2D.ParametersExtraData()
    )


class LayerCropping2DData(LayerMixinData):
    main: types.Cropping2D.ParametersMainData = types.Cropping2D.ParametersMainData()
    extra: types.Cropping2D.ParametersExtraData = types.Cropping2D.ParametersExtraData()


class LayerAttentionData(LayerMixinData):
    main: types.Attention.ParametersMainData = types.Attention.ParametersMainData()
    extra: types.Attention.ParametersExtraData = types.Attention.ParametersExtraData()


class LayerAverageData(LayerMixinData):
    main: types.Average.ParametersMainData = types.Average.ParametersMainData()
    extra: types.Average.ParametersExtraData = types.Average.ParametersExtraData()


class LayerThresholdedReLUData(LayerMixinData):
    main: types.ThresholdedReLU.ParametersMainData = (
        types.ThresholdedReLU.ParametersMainData()
    )
    extra: types.ThresholdedReLU.ParametersExtraData = (
        types.ThresholdedReLU.ParametersExtraData()
    )


class LayerRescalingData(LayerMixinData):
    main: types.Rescaling.ParametersMainData = types.Rescaling.ParametersMainData()
    extra: types.Rescaling.ParametersExtraData = types.Rescaling.ParametersExtraData()


class LayerResizingData(LayerMixinData):
    main: types.Resizing.ParametersMainData = types.Resizing.ParametersMainData()
    extra: types.Resizing.ParametersExtraData = types.Resizing.ParametersExtraData()


class LayerInceptionV3Data(LayerMixinData):
    main: types.InceptionV3.ParametersMainData = types.InceptionV3.ParametersMainData()
    extra: types.InceptionV3.ParametersExtraData = (
        types.InceptionV3.ParametersExtraData()
    )


class LayerXceptionData(LayerMixinData):
    main: types.Xception.ParametersMainData = types.Xception.ParametersMainData()
    extra: types.Xception.ParametersExtraData = types.Xception.ParametersExtraData()


class LayerVGG16Data(LayerMixinData):
    main: types.VGG16.ParametersMainData = types.VGG16.ParametersMainData()
    extra: types.VGG16.ParametersExtraData = types.VGG16.ParametersExtraData()


class LayerVGG19Data(LayerMixinData):
    main: types.VGG19.ParametersMainData = types.VGG19.ParametersMainData()
    extra: types.VGG19.ParametersExtraData = types.VGG19.ParametersExtraData()


class LayerResNet50Data(LayerMixinData):
    main: types.ResNet50.ParametersMainData = types.ResNet50.ParametersMainData()
    extra: types.ResNet50.ParametersExtraData = types.ResNet50.ParametersExtraData()


class LayerResNet101Data(LayerMixinData):
    main: types.ResNet101.ParametersMainData = types.ResNet101.ParametersMainData()
    extra: types.ResNet101.ParametersExtraData = types.ResNet101.ParametersExtraData()


class LayerResNet152Data(LayerMixinData):
    main: types.ResNet152.ParametersMainData = types.ResNet152.ParametersMainData()
    extra: types.ResNet152.ParametersExtraData = types.ResNet152.ParametersExtraData()


class LayerResNet50V2Data(LayerMixinData):
    main: types.ResNet50V2.ParametersMainData = types.ResNet50V2.ParametersMainData()
    extra: types.ResNet50V2.ParametersExtraData = types.ResNet50V2.ParametersExtraData()


class LayerResNet101V2Data(LayerMixinData):
    main: types.ResNet101V2.ParametersMainData = types.ResNet101V2.ParametersMainData()
    extra: types.ResNet101V2.ParametersExtraData = (
        types.ResNet101V2.ParametersExtraData()
    )


class LayerResNet152V2Data(LayerMixinData):
    main: types.ResNet152V2.ParametersMainData = types.ResNet152V2.ParametersMainData()
    extra: types.ResNet152V2.ParametersExtraData = (
        types.ResNet152V2.ParametersExtraData()
    )


class LayerDenseNet121Data(LayerMixinData):
    main: types.DenseNet121.ParametersMainData = types.DenseNet121.ParametersMainData()
    extra: types.DenseNet121.ParametersExtraData = (
        types.DenseNet121.ParametersExtraData()
    )


class LayerDenseNet169Data(LayerMixinData):
    main: types.DenseNet169.ParametersMainData = types.DenseNet169.ParametersMainData()
    extra: types.DenseNet169.ParametersExtraData = (
        types.DenseNet169.ParametersExtraData()
    )


class LayerDenseNet201Data(LayerMixinData):
    main: types.DenseNet201.ParametersMainData = types.DenseNet201.ParametersMainData()
    extra: types.DenseNet201.ParametersExtraData = (
        types.DenseNet201.ParametersExtraData()
    )


class LayerNASNetMobileData(LayerMixinData):
    main: types.NASNetMobile.ParametersMainData = (
        types.NASNetMobile.ParametersMainData()
    )
    extra: types.NASNetMobile.ParametersExtraData = (
        types.NASNetMobile.ParametersExtraData()
    )


class LayerNASNetLargeData(LayerMixinData):
    main: types.NASNetLarge.ParametersMainData = types.NASNetLarge.ParametersMainData()
    extra: types.NASNetLarge.ParametersExtraData = (
        types.NASNetLarge.ParametersExtraData()
    )


class LayerMobileNetV3SmallData(LayerMixinData):
    main: types.MobileNetV3Small.ParametersMainData = (
        types.MobileNetV3Small.ParametersMainData()
    )
    extra: types.MobileNetV3Small.ParametersExtraData = (
        types.MobileNetV3Small.ParametersExtraData()
    )


class LayerMobileNetV2Data(LayerMixinData):
    main: types.MobileNetV2.ParametersMainData = types.MobileNetV2.ParametersMainData()
    extra: types.MobileNetV2.ParametersExtraData = (
        types.MobileNetV2.ParametersExtraData()
    )


class LayerEfficientNetB0Data(LayerMixinData):
    main: types.EfficientNetB0.ParametersMainData = (
        types.EfficientNetB0.ParametersMainData()
    )
    extra: types.EfficientNetB0.ParametersExtraData = (
        types.EfficientNetB0.ParametersExtraData()
    )


class LayerYOLOResBlockData(LayerMixinData):
    main: types.YOLOResBlock.ParametersMainData = (
        types.YOLOResBlock.ParametersMainData()
    )
    extra: types.YOLOResBlock.ParametersExtraData = (
        types.YOLOResBlock.ParametersExtraData()
    )


class LayerYOLOv3ResBlockData(LayerMixinData):
    main: types.YOLOv3ResBlock.ParametersMainData = (
        types.YOLOv3ResBlock.ParametersMainData()
    )
    extra: types.YOLOv3ResBlock.ParametersExtraData = (
        types.YOLOv3ResBlock.ParametersExtraData()
    )


class LayerYOLOConvBlockData(LayerMixinData):
    main: types.YOLOConvBlock.ParametersMainData = (
        types.YOLOConvBlock.ParametersMainData()
    )
    extra: types.YOLOConvBlock.ParametersExtraData = (
        types.YOLOConvBlock.ParametersExtraData()
    )


class LayerVAEBlockData(LayerMixinData):
    main: types.VAEBlock.ParametersMainData = types.VAEBlock.ParametersMainData()
    extra: types.VAEBlock.ParametersExtraData = types.VAEBlock.ParametersExtraData()


class LayerCustomBlockData(LayerMixinData):
    main: types.CustomBlock.ParametersMainData = types.CustomBlock.ParametersMainData()
    extra: types.CustomBlock.ParametersExtraData = (
        types.CustomBlock.ParametersExtraData()
    )


class LayerSpaceToDepthData(LayerMixinData):
    main: types.SpaceToDepth.ParametersMainData = (
        types.SpaceToDepth.ParametersMainData()
    )
    extra: types.SpaceToDepth.ParametersExtraData = (
        types.SpaceToDepth.ParametersExtraData()
    )


class LayerDepthToSpaceData(LayerMixinData):
    main: types.DepthToSpace.ParametersMainData = (
        types.DepthToSpace.ParametersMainData()
    )
    extra: types.DepthToSpace.ParametersExtraData = (
        types.DepthToSpace.ParametersExtraData()
    )


# class LayerDarkNetConvolutionalData(LayerMixinData):
#     main: types.DarkNetConvolutional.ParametersMainData = (
#         types.DarkNetConvolutional.ParametersMainData()
#     )
#     extra: types.DarkNetConvolutional.ParametersExtraData = (
#         types.DarkNetConvolutional.ParametersExtraData()
#     )
#
#
# class LayerDarkNetUpsampleData(LayerMixinData):
#     main: types.DarkNetUpsample.ParametersMainData = (
#         types.DarkNetUpsample.ParametersMainData()
#     )
#     extra: types.DarkNetUpsample.ParametersExtraData = (
#         types.DarkNetUpsample.ParametersExtraData()
#     )
#
#
# class LayerDarkNetResBlockData(LayerMixinData):
#     main: types.DarkNetResBlock.ParametersMainData = (
#         types.DarkNetResBlock.ParametersMainData()
#     )
#     extra: types.DarkNetResBlock.ParametersExtraData = (
#         types.DarkNetResBlock.ParametersExtraData()
#     )
#
#
# class LayerDarkNetBatchNormalizationData(LayerMixinData):
#     main: types.DarkNetBatchNormalization.ParametersMainData = (
#         types.DarkNetBatchNormalization.ParametersMainData()
#     )
#     extra: types.DarkNetBatchNormalization.ParametersExtraData = (
#         types.DarkNetBatchNormalization.ParametersExtraData()
#     )


class LayerCONVBlockData(LayerMixinData):
    main: types.CONVBlock.ParametersMainData = types.CONVBlock.ParametersMainData()
    extra: types.CONVBlock.ParametersExtraData = types.CONVBlock.ParametersExtraData()


class LayerPSPBlock1DData(LayerMixinData):
    main: types.PSPBlock1D.ParametersMainData = types.PSPBlock1D.ParametersMainData()
    extra: types.PSPBlock1D.ParametersExtraData = types.PSPBlock1D.ParametersExtraData()


class LayerPSPBlock2DData(LayerMixinData):
    main: types.PSPBlock2D.ParametersMainData = types.PSPBlock2D.ParametersMainData()
    extra: types.PSPBlock2D.ParametersExtraData = types.PSPBlock2D.ParametersExtraData()


class LayerPSPBlock3DData(LayerMixinData):
    main: types.PSPBlock3D.ParametersMainData = types.PSPBlock3D.ParametersMainData()
    extra: types.PSPBlock3D.ParametersExtraData = types.PSPBlock3D.ParametersExtraData()


class LayerUNETBlock2DData(LayerMixinData):
    main: types.UNETBlock2D.ParametersMainData = types.UNETBlock2D.ParametersMainData()
    extra: types.UNETBlock2D.ParametersExtraData = (
        types.UNETBlock2D.ParametersExtraData()
    )


class LayerUNETBlock1DData(LayerMixinData):
    main: types.UNETBlock1D.ParametersMainData = types.UNETBlock1D.ParametersMainData()
    extra: types.UNETBlock1D.ParametersExtraData = (
        types.UNETBlock1D.ParametersExtraData()
    )


class LayerUNETBlock3DData(LayerMixinData):
    main: types.UNETBlock3D.ParametersMainData = types.UNETBlock3D.ParametersMainData()
    extra: types.UNETBlock3D.ParametersExtraData = (
        types.UNETBlock3D.ParametersExtraData()
    )


class LayerConditionalMergeLayerData(LayerMixinData):
    main: types.ConditionalMergeLayer.ParametersMainData = types.ConditionalMergeLayer.ParametersMainData()
    extra: types.ConditionalMergeLayer.ParametersExtraData = (
        types.ConditionalMergeLayer.ParametersExtraData()
    )


class LayerResnetBlock2DData(LayerMixinData):
    main: types.ResnetBlock2D.ParametersMainData = types.ResnetBlock2D.ParametersMainData()
    extra: types.ResnetBlock2D.ParametersExtraData = (types.ResnetBlock2D.ParametersExtraData())


class LayerRGBNormalizationData(LayerMixinData):
    main: types.RGBNormalization.ParametersMainData = types.RGBNormalization.ParametersMainData()
    extra: types.RGBNormalization.ParametersExtraData = (types.RGBNormalization.ParametersExtraData())


class LayerTransformerData(LayerMixinData):
    main: types.Transformer.ParametersMainData = types.Transformer.ParametersMainData()
    extra: types.Transformer.ParametersExtraData = (types.Transformer.ParametersExtraData())


class LayerFNetTransformerData(LayerMixinData):
    main: types.FNetTransformer.ParametersMainData = types.FNetTransformer.ParametersMainData()
    extra: types.FNetTransformer.ParametersExtraData = (types.FNetTransformer.ParametersExtraData())


class LayerPretrainedBERTData(LayerMixinData):
    main: types.PretrainedBERT.ParametersMainData = (
        types.PretrainedBERT.ParametersMainData()
    )
    extra: types.PretrainedBERT.ParametersExtraData = (
        types.PretrainedBERT.ParametersExtraData()
    )


class LayerPretrainedYOLOData(LayerMixinData):
    main: types.PretrainedYOLO.ParametersMainData = (
        types.PretrainedYOLO.ParametersMainData()
    )
    extra: types.PretrainedYOLO.ParametersExtraData = (
        types.PretrainedYOLO.ParametersExtraData()
    )
    weight_path: Optional[FilePath]

    @validator("weight_path", always=True)
    def _validate_weight_path(cls, value):
        if not value:
            value = None
        return value

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"weight_path"}})
        return super().dict(**kwargs)

    def weight_load(self):
        os.makedirs(settings.WEIGHT_PATH, exist_ok=True)
        value = None
        if self.main.use_weights:
            weight_filename = WEIGHT_FILES.get(self.main.version)
            value = Path(settings.WEIGHT_PATH, weight_filename)
            if not value.is_file():
                filepath = progress_utils.download(
                    "weight_load",
                    "Загрузка весов `{weight_filename}`",
                    f"{settings.WEIGHT_STORAGE_URL}{weight_filename}",
                )
                shutil.move(filepath, value)
        self.weight_path = value


Layer = Enum(
    "Layer",
    dict(map(lambda item: (item.name, f"Layer{item.name}Data"), list(LayerTypeChoice))),
    type=str,
)
