"""
## Структура данных для параметров слоев
"""

import sys

from enum import Enum

from ..mixins import BaseMixinData
from .extra import LayerTypeChoice
from . import layers_parameters as lps


class ParametersTypeDefaultData(BaseMixinData):
    pass


class ParametersTypeMixinData(BaseMixinData):
    main: ParametersTypeDefaultData = ParametersTypeDefaultData()
    extra: ParametersTypeDefaultData = ParametersTypeDefaultData()


class ParametersTypeInputData(ParametersTypeMixinData):
    main: lps.Input.ParametersMainData = lps.Input.ParametersMainData()
    extra: lps.Input.ParametersExtraData = lps.Input.ParametersExtraData()


class ParametersTypeConv1DData(ParametersTypeMixinData):
    main: lps.Conv1D.ParametersMainData = lps.Conv1D.ParametersMainData()
    extra: lps.Conv1D.ParametersExtraData = lps.Conv1D.ParametersExtraData()


class ParametersTypeConv2DData(ParametersTypeMixinData):
    main: lps.Conv2D.ParametersMainData = lps.Conv2D.ParametersMainData()
    extra: lps.Conv2D.ParametersExtraData = lps.Conv2D.ParametersExtraData()


class ParametersTypeConv3DData(ParametersTypeMixinData):
    main: lps.Conv3D.ParametersMainData = lps.Conv3D.ParametersMainData()
    extra: lps.Conv3D.ParametersExtraData = lps.Conv3D.ParametersExtraData()


class ParametersTypeConv1DTransposeData(ParametersTypeMixinData):
    main: lps.Conv1DTranspose.ParametersMainData = (
        lps.Conv1DTranspose.ParametersMainData()
    )
    extra: lps.Conv1DTranspose.ParametersExtraData = (
        lps.Conv1DTranspose.ParametersExtraData()
    )


class ParametersTypeConv2DTransposeData(ParametersTypeMixinData):
    main: lps.Conv2DTranspose.ParametersMainData = (
        lps.Conv2DTranspose.ParametersMainData()
    )
    extra: lps.Conv2DTranspose.ParametersExtraData = (
        lps.Conv2DTranspose.ParametersExtraData()
    )


class ParametersTypeSeparableConv1DData(ParametersTypeMixinData):
    main: lps.SeparableConv1D.ParametersMainData = (
        lps.SeparableConv1D.ParametersMainData()
    )
    extra: lps.SeparableConv1D.ParametersExtraData = (
        lps.SeparableConv1D.ParametersExtraData()
    )


class ParametersTypeSeparableConv2DData(ParametersTypeMixinData):
    main: lps.SeparableConv2D.ParametersMainData = (
        lps.SeparableConv2D.ParametersMainData()
    )
    extra: lps.SeparableConv2D.ParametersExtraData = (
        lps.SeparableConv2D.ParametersExtraData()
    )


class ParametersTypeDepthwiseConv2DData(ParametersTypeMixinData):
    main: lps.DepthwiseConv2D.ParametersMainData = (
        lps.DepthwiseConv2D.ParametersMainData()
    )
    extra: lps.DepthwiseConv2D.ParametersExtraData = (
        lps.DepthwiseConv2D.ParametersExtraData()
    )


class ParametersTypeMaxPooling1DData(ParametersTypeMixinData):
    main: lps.MaxPooling1D.ParametersMainData = lps.MaxPooling1D.ParametersMainData()
    extra: lps.MaxPooling1D.ParametersExtraData = lps.MaxPooling1D.ParametersExtraData()


class ParametersTypeMaxPooling2DData(ParametersTypeMixinData):
    main: lps.MaxPooling2D.ParametersMainData = lps.MaxPooling2D.ParametersMainData()
    extra: lps.MaxPooling2D.ParametersExtraData = lps.MaxPooling2D.ParametersExtraData()


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


class ParametersTypeUpSampling1DData(ParametersTypeMixinData):
    main: lps.UpSampling1D.ParametersMainData = lps.UpSampling1D.ParametersMainData()
    extra: lps.UpSampling1D.ParametersExtraData = lps.UpSampling1D.ParametersExtraData()


class ParametersTypeUpSampling2DData(ParametersTypeMixinData):
    main: lps.UpSampling2D.ParametersMainData = lps.UpSampling2D.ParametersMainData()
    extra: lps.UpSampling2D.ParametersExtraData = lps.UpSampling2D.ParametersExtraData()


class ParametersTypeLeakyReLUData(ParametersTypeMixinData):
    main: lps.LeakyReLU.ParametersMainData = lps.LeakyReLU.ParametersMainData()
    extra: lps.LeakyReLU.ParametersExtraData = lps.LeakyReLU.ParametersExtraData()


class ParametersTypeDropoutData(ParametersTypeMixinData):
    main: lps.Dropout.ParametersMainData = lps.Dropout.ParametersMainData()
    extra: lps.Dropout.ParametersExtraData = lps.Dropout.ParametersExtraData()


class ParametersTypeDenseData(ParametersTypeMixinData):
    main: lps.Dense.ParametersMainData = lps.Dense.ParametersMainData()
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


class ParametersTypesigmoidData(ParametersTypeMixinData):
    main: lps.sigmoid.ParametersMainData = lps.sigmoid.ParametersMainData()
    extra: lps.sigmoid.ParametersExtraData = lps.sigmoid.ParametersExtraData()


class ParametersTypesoftmaxData(ParametersTypeMixinData):
    main: lps.softmax.ParametersMainData = lps.softmax.ParametersMainData()
    extra: lps.softmax.ParametersExtraData = lps.softmax.ParametersExtraData()


class ParametersTypetanhData(ParametersTypeMixinData):
    main: lps.tanh.ParametersMainData = lps.tanh.ParametersMainData()
    extra: lps.tanh.ParametersExtraData = lps.tanh.ParametersExtraData()


class ParametersTypereluData(ParametersTypeMixinData):
    main: lps.relu.ParametersMainData = lps.relu.ParametersMainData()
    extra: lps.relu.ParametersExtraData = lps.relu.ParametersExtraData()


class ParametersTypeeluData(ParametersTypeMixinData):
    main: lps.elu.ParametersMainData = lps.elu.ParametersMainData()
    extra: lps.elu.ParametersExtraData = lps.elu.ParametersExtraData()


class ParametersTypeseluData(ParametersTypeMixinData):
    main: lps.selu.ParametersMainData = lps.selu.ParametersMainData()
    extra: lps.selu.ParametersExtraData = lps.selu.ParametersExtraData()


class ParametersTypePReLUData(ParametersTypeMixinData):
    main: lps.PReLU.ParametersMainData = lps.PReLU.ParametersMainData()
    extra: lps.PReLU.ParametersExtraData = lps.PReLU.ParametersExtraData()


class ParametersTypeGlobalMaxPooling1DData(ParametersTypeMixinData):
    main: lps.GlobalMaxPooling1D.ParametersMainData = (
        lps.GlobalMaxPooling1D.ParametersMainData()
    )
    extra: lps.GlobalMaxPooling1D.ParametersExtraData = (
        lps.GlobalMaxPooling1D.ParametersExtraData()
    )


class ParametersTypeGlobalMaxPooling2DData(ParametersTypeMixinData):
    main: lps.GlobalMaxPooling2D.ParametersMainData = (
        lps.GlobalMaxPooling2D.ParametersMainData()
    )
    extra: lps.GlobalMaxPooling2D.ParametersExtraData = (
        lps.GlobalMaxPooling2D.ParametersExtraData()
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


class ParametersTypeGRUData(ParametersTypeMixinData):
    main: lps.GRU.ParametersMainData = lps.GRU.ParametersMainData()
    extra: lps.GRU.ParametersExtraData = lps.GRU.ParametersExtraData()


class ParametersTypeLSTMData(ParametersTypeMixinData):
    main: lps.LSTM.ParametersMainData = lps.LSTM.ParametersMainData()
    extra: lps.LSTM.ParametersExtraData = lps.LSTM.ParametersExtraData()


class ParametersTypeEmbeddingData(ParametersTypeMixinData):
    main: lps.Embedding.ParametersMainData = lps.Embedding.ParametersMainData()
    extra: lps.Embedding.ParametersExtraData = lps.Embedding.ParametersExtraData()


class ParametersTypeRepeatVectorData(ParametersTypeMixinData):
    main: lps.RepeatVector.ParametersMainData = lps.RepeatVector.ParametersMainData()
    extra: lps.RepeatVector.ParametersExtraData = lps.RepeatVector.ParametersExtraData()


class ParametersTypeBatchNormalizationData(ParametersTypeMixinData):
    main: lps.BatchNormalization.ParametersMainData = (
        lps.BatchNormalization.ParametersMainData()
    )
    extra: lps.BatchNormalization.ParametersExtraData = (
        lps.BatchNormalization.ParametersExtraData()
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
