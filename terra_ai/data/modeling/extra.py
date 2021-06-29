"""
## Дополнительные структуры данных для моделей
"""

from enum import Enum


class LayerGroupChoice(str, Enum):
    input = "input"
    middle = "middle"
    output = "output"


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
    MaxPooling1D = "MaxPooling1D"
    MaxPooling2D = "MaxPooling2D"
    AveragePooling1D = "AveragePooling1D"
    AveragePooling2D = "AveragePooling2D"
    AveragePooling3D = "AveragePooling3D"
    UpSampling1D = "UpSampling1D"
    UpSampling2D = "UpSampling2D"
    LeakyReLU = "LeakyReLU"
    Dropout = "Dropout"
    Dense = "Dense"
    Add = "Add"
    Multiply = "Multiply"
    Flatten = "Flatten"
    Concatenate = "Concatenate"
    Reshape = "Reshape"
    sigmoid = "sigmoid"
    softmax = "softmax"
    tanh = "tanh"
    relu = "relu"
    ELU = "ELU"
    selu = "selu"
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
