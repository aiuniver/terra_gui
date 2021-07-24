"""
## Дополнительные структуры данных для моделей
"""

from enum import Enum


class ReferenceTypeChoice(str, Enum):
    block = "block"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ReferenceTypeChoice))


class LayerGroupChoice(str, Enum):
    input = "input"
    middle = "middle"
    output = "output"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, LayerGroupChoice))


class LayerTypeChoice(str, Enum):
    Input = "Input"
    Conv1D = "Conv1D"
    Conv2D = "Conv2D"
    Conv3D = "Conv3D"
    Conv1DTranspose = "Conv1DTranspose"
    Conv2DTranspose = "Conv2DTranspose"
    Conv3DTranspose = "Conv3DTranspose"
    SeparableConv1D = "SeparableConv1D"
    SeparableConv2D = "SeparableConv2D"
    DepthwiseConv2D = "DepthwiseConv2D"
    MaxPool1D = "MaxPool1D"
    MaxPool2D = "MaxPool2D"
    MaxPool3D = "MaxPool3D"
    AveragePooling1D = "AveragePooling1D"
    AveragePooling2D = "AveragePooling2D"
    AveragePooling3D = "AveragePooling3D"
    UpSampling1D = "UpSampling1D"
    UpSampling2D = "UpSampling2D"
    UpSampling3D = "UpSampling3D"
    LeakyReLU = "LeakyReLU"
    Dropout = "Dropout"
    Dense = "Dense"
    Add = "Add"
    Multiply = "Multiply"
    Flatten = "Flatten"
    Concatenate = "Concatenate"
    Reshape = "Reshape"
    Activation = "Activation"
    Softmax = "Softmax"
    ReLU = "ReLU"
    ELU = "ELU"
    PReLU = "PReLU"
    GlobalMaxPool1D = "GlobalMaxPool1D"
    GlobalMaxPool2D = "GlobalMaxPool2D"
    GlobalMaxPool3D = "GlobalMaxPool3D"
    GlobalAveragePooling1D = "GlobalAveragePooling1D"
    GlobalAveragePooling2D = "GlobalAveragePooling2D"
    GlobalAveragePooling3D = "GlobalAveragePooling3D"
    GRU = "GRU"
    LSTM = "LSTM"
    Embedding = "Embedding"
    RepeatVector = "RepeatVector"
    BatchNormalization = "BatchNormalization"
    Mish = "Mish"
    InstanceNormalization = "InstanceNormalization"
    ZeroPadding2D = "ZeroPadding2D"
    Cropping2D = "Cropping2D"
    Attention = "Attention"
    Normalization = "Normalization"
    Average = "Average"
    ThresholdedReLU = "ThresholdedReLU"
    Rescaling = "Rescaling"
    Resizing = "Resizing"
    InceptionV3 = "InceptionV3"
    Xception = "Xception"
    VGG16 = "VGG16"
    ResNet50 = "ResNet50"
    CustomUNETBlock = "CustomUNETBlock"
    YOLOResBlock = "YOLOResBlock"
    YOLOConvBlock = "YOLOConvBlock"
    VAEBlock = "VAEBlock"
    CustomBlock = "CustomBlock"
    SpaceToDepth = "SpaceToDepth"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, LayerTypeChoice))
