"""
## Структура данных для параметров слоев
"""

import sys

from enum import Enum

import tensorflow

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.modeling.layers_parameters.extra import ActivationChoice
from terra_ai.data.modeling import layers_parameters as lps


class ParametersTypeDefaultData(BaseMixinData):
    pass


class ParametersTypeMainDefaultData(ParametersTypeDefaultData):
    pass


class ParametersTypeExtraDefaultData(ParametersTypeDefaultData):
    pass


class ParametersTypeMixinData(BaseMixinData):
    main: ParametersTypeMainDefaultData = ParametersTypeMainDefaultData()
    extra: ParametersTypeExtraDefaultData = ParametersTypeExtraDefaultData()
    module: str = 'tensorflow.keras.layers'


class ParametersTypeInputData(ParametersTypeMixinData):
    main: lps.Input.ParametersMainData = lps.Input.ParametersMainData()
    extra: lps.Input.ParametersExtraData = lps.Input.ParametersExtraData()


class ParametersTypeConv1DData(ParametersTypeMixinData):
    main: lps.Conv1D.ParametersMainData = lps.Conv1D.ParametersMainData(
        filters=32, kernel_size=5
    )
    extra: lps.Conv1D.ParametersExtraData = lps.Conv1D.ParametersExtraData()


class ParametersTypeConv2DData(ParametersTypeMixinData):
    main: lps.Conv2D.ParametersMainData = lps.Conv2D.ParametersMainData(
        filters=32, kernel_size=(3, 3)
    )
    extra: lps.Conv2D.ParametersExtraData = lps.Conv2D.ParametersExtraData()


class ParametersTypeConv3DData(ParametersTypeMixinData):
    main: lps.Conv3D.ParametersMainData = lps.Conv3D.ParametersMainData(
        filters=32, kernel_size=(3, 3, 3)
    )
    extra: lps.Conv3D.ParametersExtraData = lps.Conv3D.ParametersExtraData()


class ParametersTypeConv1DTransposeData(ParametersTypeMixinData):
    main: lps.Conv1DTranspose.ParametersMainData = (
        lps.Conv1DTranspose.ParametersMainData(filters=32, kernel_size=1)
    )
    extra: lps.Conv1DTranspose.ParametersExtraData = (
        lps.Conv1DTranspose.ParametersExtraData()
    )


class ParametersTypeConv2DTransposeData(ParametersTypeMixinData):
    main: lps.Conv2DTranspose.ParametersMainData = (
        lps.Conv2DTranspose.ParametersMainData(filters=32, kernel_size=(1, 1))
    )
    extra: lps.Conv2DTranspose.ParametersExtraData = (
        lps.Conv2DTranspose.ParametersExtraData()
    )


class ParametersTypeConv3DTransposeData(ParametersTypeMixinData):
    main: lps.Conv3DTranspose.ParametersMainData = (
        lps.Conv3DTranspose.ParametersMainData(filters=32, kernel_size=(1, 1, 1))
    )
    extra: lps.Conv3DTranspose.ParametersExtraData = (
        lps.Conv3DTranspose.ParametersExtraData()
    )


class ParametersTypeSeparableConv1DData(ParametersTypeMixinData):
    main: lps.SeparableConv1D.ParametersMainData = (
        lps.SeparableConv1D.ParametersMainData(filters=32, kernel_size=1)
    )
    extra: lps.SeparableConv1D.ParametersExtraData = (
        lps.SeparableConv1D.ParametersExtraData()
    )


class ParametersTypeSeparableConv2DData(ParametersTypeMixinData):
    main: lps.SeparableConv2D.ParametersMainData = (
        lps.SeparableConv2D.ParametersMainData(filters=32, kernel_size=(1, 1))
    )
    extra: lps.SeparableConv2D.ParametersExtraData = (
        lps.SeparableConv2D.ParametersExtraData()
    )


class ParametersTypeDepthwiseConv2DData(ParametersTypeMixinData):
    main: lps.DepthwiseConv2D.ParametersMainData = (
        lps.DepthwiseConv2D.ParametersMainData(kernel_size=(1, 1))
    )
    extra: lps.DepthwiseConv2D.ParametersExtraData = (
        lps.DepthwiseConv2D.ParametersExtraData()
    )


class ParametersTypeMaxPool1DData(ParametersTypeMixinData):
    main: lps.MaxPool1D.ParametersMainData = lps.MaxPool1D.ParametersMainData()
    extra: lps.MaxPool1D.ParametersExtraData = lps.MaxPool1D.ParametersExtraData()


class ParametersTypeMaxPool2DData(ParametersTypeMixinData):
    main: lps.MaxPool2D.ParametersMainData = lps.MaxPool2D.ParametersMainData()
    extra: lps.MaxPool2D.ParametersExtraData = lps.MaxPool2D.ParametersExtraData()


class ParametersTypeMaxPool3DData(ParametersTypeMixinData):
    main: lps.MaxPool3D.ParametersMainData = lps.MaxPool3D.ParametersMainData()
    extra: lps.MaxPool3D.ParametersExtraData = lps.MaxPool3D.ParametersExtraData()


class ParametersTypeAveragePooling1DData(ParametersTypeMixinData):
    main: lps.AveragePooling1D.ParametersMainData = (
        lps.AveragePooling1D.ParametersMainData()
    )
    extra: lps.AveragePooling1D.ParametersExtraData = (
        lps.AveragePooling1D.ParametersExtraData()
    )


class ParametersTypeAveragePooling2DData(ParametersTypeMixinData):
    main: lps.AveragePooling2D.ParametersMainData = (
        lps.AveragePooling2D.ParametersMainData()
    )
    extra: lps.AveragePooling2D.ParametersExtraData = (
        lps.AveragePooling2D.ParametersExtraData()
    )


class ParametersTypeAveragePooling3DData(ParametersTypeMixinData):
    main: lps.AveragePooling3D.ParametersMainData = (
        lps.AveragePooling3D.ParametersMainData()
    )
    extra: lps.AveragePooling3D.ParametersExtraData = (
        lps.AveragePooling3D.ParametersExtraData()
    )


class ParametersTypeUpSampling1DData(ParametersTypeMixinData):
    main: lps.UpSampling1D.ParametersMainData = lps.UpSampling1D.ParametersMainData()
    extra: lps.UpSampling1D.ParametersExtraData = lps.UpSampling1D.ParametersExtraData()


class ParametersTypeUpSampling2DData(ParametersTypeMixinData):
    main: lps.UpSampling2D.ParametersMainData = lps.UpSampling2D.ParametersMainData()
    extra: lps.UpSampling2D.ParametersExtraData = lps.UpSampling2D.ParametersExtraData()


class ParametersTypeUpSampling3DData(ParametersTypeMixinData):
    main: lps.UpSampling3D.ParametersMainData = lps.UpSampling3D.ParametersMainData()
    extra: lps.UpSampling3D.ParametersExtraData = lps.UpSampling3D.ParametersExtraData()


class ParametersTypeLeakyReLUData(ParametersTypeMixinData):
    main: lps.LeakyReLU.ParametersMainData = lps.LeakyReLU.ParametersMainData()
    extra: lps.LeakyReLU.ParametersExtraData = lps.LeakyReLU.ParametersExtraData()


class ParametersTypeDropoutData(ParametersTypeMixinData):
    main: lps.Dropout.ParametersMainData = lps.Dropout.ParametersMainData(rate=0.1)
    extra: lps.Dropout.ParametersExtraData = lps.Dropout.ParametersExtraData()


class ParametersTypeDenseData(ParametersTypeMixinData):
    main: lps.Dense.ParametersMainData = lps.Dense.ParametersMainData(units=32)
    extra: lps.Dense.ParametersExtraData = lps.Dense.ParametersExtraData()


class ParametersTypeAddData(ParametersTypeMixinData):
    main: lps.Add.ParametersMainData = lps.Add.ParametersMainData()
    extra: lps.Add.ParametersExtraData = lps.Add.ParametersExtraData()


class ParametersTypeMultiplyData(ParametersTypeMixinData):
    main: lps.Multiply.ParametersMainData = lps.Multiply.ParametersMainData()
    extra: lps.Multiply.ParametersExtraData = lps.Multiply.ParametersExtraData()


class ParametersTypeFlattenData(ParametersTypeMixinData):
    main: lps.Flatten.ParametersMainData = lps.Flatten.ParametersMainData()
    extra: lps.Flatten.ParametersExtraData = lps.Flatten.ParametersExtraData()


class ParametersTypeConcatenateData(ParametersTypeMixinData):
    main: lps.Concatenate.ParametersMainData = lps.Concatenate.ParametersMainData()
    extra: lps.Concatenate.ParametersExtraData = lps.Concatenate.ParametersExtraData()


class ParametersTypeReshapeData(ParametersTypeMixinData):
    main: lps.Reshape.ParametersMainData = lps.Reshape.ParametersMainData()
    extra: lps.Reshape.ParametersExtraData = lps.Reshape.ParametersExtraData()


class ParametersTypeSoftmaxData(ParametersTypeMixinData):
    main: lps.Softmax.ParametersMainData = lps.Softmax.ParametersMainData()
    extra: lps.Softmax.ParametersExtraData = lps.Softmax.ParametersExtraData()


class ParametersTypeReLUData(ParametersTypeMixinData):
    main: lps.ReLU.ParametersMainData = lps.ReLU.ParametersMainData()
    extra: lps.ReLU.ParametersExtraData = lps.ReLU.ParametersExtraData()


class ParametersTypeELUData(ParametersTypeMixinData):
    main: lps.ELU.ParametersMainData = lps.ELU.ParametersMainData()
    extra: lps.ELU.ParametersExtraData = lps.ELU.ParametersExtraData()


class ParametersTypeActivationData(ParametersTypeMixinData):
    main: lps.Activation.ParametersMainData = lps.Activation.ParametersMainData(
        activation=ActivationChoice.relu
    )
    extra: lps.Activation.ParametersExtraData = lps.Activation.ParametersExtraData()


class ParametersTypePReLUData(ParametersTypeMixinData):
    main: lps.PReLU.ParametersMainData = lps.PReLU.ParametersMainData()
    extra: lps.PReLU.ParametersExtraData = lps.PReLU.ParametersExtraData()


class ParametersTypeGlobalMaxPool1DData(ParametersTypeMixinData):
    main: lps.GlobalMaxPool1D.ParametersMainData = (
        lps.GlobalMaxPool1D.ParametersMainData()
    )
    extra: lps.GlobalMaxPool1D.ParametersExtraData = (
        lps.GlobalMaxPool1D.ParametersExtraData()
    )


class ParametersTypeGlobalMaxPool2DData(ParametersTypeMixinData):
    main: lps.GlobalMaxPool2D.ParametersMainData = (
        lps.GlobalMaxPool2D.ParametersMainData()
    )
    extra: lps.GlobalMaxPool2D.ParametersExtraData = (
        lps.GlobalMaxPool2D.ParametersExtraData()
    )


class ParametersTypeGlobalMaxPool3DData(ParametersTypeMixinData):
    main: lps.GlobalMaxPool3D.ParametersMainData = (
        lps.GlobalMaxPool3D.ParametersMainData()
    )
    extra: lps.GlobalMaxPool3D.ParametersExtraData = (
        lps.GlobalMaxPool3D.ParametersExtraData()
    )


class ParametersTypeGlobalAveragePooling1DData(ParametersTypeMixinData):
    main: lps.GlobalAveragePooling1D.ParametersMainData = (
        lps.GlobalAveragePooling1D.ParametersMainData()
    )
    extra: lps.GlobalAveragePooling1D.ParametersExtraData = (
        lps.GlobalAveragePooling1D.ParametersExtraData()
    )


class ParametersTypeGlobalAveragePooling2DData(ParametersTypeMixinData):
    main: lps.GlobalAveragePooling2D.ParametersMainData = (
        lps.GlobalAveragePooling2D.ParametersMainData()
    )
    extra: lps.GlobalAveragePooling2D.ParametersExtraData = (
        lps.GlobalAveragePooling2D.ParametersExtraData()
    )


class ParametersTypeGlobalAveragePooling3DData(ParametersTypeMixinData):
    main: lps.GlobalAveragePooling3D.ParametersMainData = (
        lps.GlobalAveragePooling3D.ParametersMainData()
    )
    extra: lps.GlobalAveragePooling3D.ParametersExtraData = (
        lps.GlobalAveragePooling3D.ParametersExtraData()
    )


class ParametersTypeGRUData(ParametersTypeMixinData):
    main: lps.GRU.ParametersMainData = lps.GRU.ParametersMainData(units=32)
    extra: lps.GRU.ParametersExtraData = lps.GRU.ParametersExtraData()


class ParametersTypeLSTMData(ParametersTypeMixinData):
    main: lps.LSTM.ParametersMainData = lps.LSTM.ParametersMainData(units=32)
    extra: lps.LSTM.ParametersExtraData = lps.LSTM.ParametersExtraData()


class ParametersTypeEmbeddingData(ParametersTypeMixinData):
    main: lps.Embedding.ParametersMainData = lps.Embedding.ParametersMainData(
        input_dim=1000, output_dim=64
    )
    extra: lps.Embedding.ParametersExtraData = lps.Embedding.ParametersExtraData()


class ParametersTypeRepeatVectorData(ParametersTypeMixinData):
    main: lps.RepeatVector.ParametersMainData = lps.RepeatVector.ParametersMainData()
    extra: lps.RepeatVector.ParametersExtraData = lps.RepeatVector.ParametersExtraData(
        n=8
    )


class ParametersTypeBatchNormalizationData(ParametersTypeMixinData):
    main: lps.BatchNormalization.ParametersMainData = (
        lps.BatchNormalization.ParametersMainData()
    )
    extra: lps.BatchNormalization.ParametersExtraData = (
        lps.BatchNormalization.ParametersExtraData()
    )

class ParametersTypeMishData(ParametersTypeMixinData):
    main: lps.Mish.ParametersMainData = (
        lps.Mish.ParametersMainData()
    )
    extra: lps.Mish.ParametersExtraData = (
        lps.Mish.ParametersExtraData()
    )

class ParametersTypeInstanceNormalizationData(ParametersTypeMixinData):
    main: lps.InstanceNormalization.ParametersMainData = (
        lps.InstanceNormalization.ParametersMainData()
    )
    extra: lps.InstanceNormalization.ParametersExtraData = (
        lps.InstanceNormalization.ParametersExtraData()
    )

class ParametersTypeZeroPadding2DData(ParametersTypeMixinData):
        main: lps.ZeroPadding2D.ParametersMainData = (
            lps.ZeroPadding2D.ParametersMainData()
        )
        extra: lps.ZeroPadding2D.ParametersExtraData = (
            lps.ZeroPadding2D.ParametersExtraData()
        )

class ParametersTypeCropping2DData(ParametersTypeMixinData):
    main: lps.Cropping2D.ParametersMainData = (
        lps.Cropping2D.ParametersMainData()
    )
    extra: lps.Cropping2D.ParametersExtraData = (
        lps.Cropping2D.ParametersExtraData()
    )



ParametersType = Enum(
    "ParametersType",
    dict(map(lambda item: (item, f"ParametersType{item}Data"), list(LayerTypeChoice))),
    type=str,
)
"""
Список возможных типов параметров слоя
"""


ParametersTypeUnion = tuple(
    map(lambda item: getattr(sys.modules.get(__name__), item), ParametersType)
)
"""
Список возможных типов параметров данных для слоя
"""
