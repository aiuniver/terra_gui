from enum import Enum
from datetime import datetime

from terra_ai.data.datasets.extra import (
    DatasetGroupChoice,
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerEncodingChoice,
)
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.modeling.layers.extra import ActivationChoice
from terra_ai.data.training.extra import ArchitectureChoice


class TagsAlias(dict, Enum):
    image = {"alias": "image", "name": "Image"}
    video = {"alias": "video", "name": "Video"}
    classification = {"alias": "classification", "name": "Classification"}
    tensorflow_keras = {"alias": "tensorflow_keras", "name": "Tensorflow.keras"}
    text = {"alias": "text", "name": "Text"}
    english = {"alias": "english", "name": "English"}
    russian = {"alias": "russian", "name": "Russian"}
    regression = {"alias": "regression", "name": "Regression"}
    timeseries = {"alias": "timeseries", "name": "Timeseries"}
    timeseriestrend = {"alias": "timeseriestrend", "name": "TimeseriesTrend"}
    terra_ai = {"alias": "terra_ai", "name": "Terra AI"}
    object_detection = {"alias": "object_detection", "name": "Object detection"}
    segmentation = {"alias": "segmentation", "name": "Segmentation"}
    text_segmentation = {"alias": "text_segmentation", "name": "Text Segmentation"}
    audio = {"alias": "audio", "name": "Audio"}
    smart_home = {"alias": "smart_home", "name": "Smart home"}
    trading = {"alias": "trading", "name": "Trading"}
    tracker = {"alias": "tracker", "name": "Tracker"}
    text_to_speech = {"alias": "text_to_speech", "name": "Text-to-Speech"}
    speech_to_text = {"alias": "speech_to_text", "name": "Speech-to-Text"}
    gan = {"alias": "gan", "name": "GAN"}
    cgan = {"alias": "cgan", "name": "CGAN"}


class Tags(str, Enum):
    image = "Image"
    video = "Video"
    classification = "Classification"
    tensorflow_keras = "Tensorflow.keras"
    text = "Text"
    english = "English"
    russian = "Russian"
    regression = "Regression"
    timeseries = "Timeseries"
    timeseriestrend = "TimeseriesTrend"
    terra_ai = "Terra AI"
    object_detection = "Object detection"
    segmentation = "Segmentation"
    text_segmentation = "Text Segmentation"
    audio = "Audio"
    smart_home = "Smart home"
    trading = "Trading"
    tracker = "Tracker"
    text_to_speech = "Text-to-Speech"
    speech_to_text = "Speech-to-Text"
    gan = "GAN"
    cgan = "CGAN"


OutputLayersDefaults = {
    LayerOutputTypeChoice.Classification: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.softmax.value,
                }
            },
        },
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.softmax.value,
                }
            },
        },
    },
    LayerOutputTypeChoice.Segmentation: {
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.softmax.value,
                }
            },
        },
        "2D": {
            "type": LayerTypeChoice.Conv2D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.softmax.value,
                }
            },
        },
        "3D": {
            "type": LayerTypeChoice.Conv3D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.softmax.value,
                }
            },
        },
    },
    LayerOutputTypeChoice.TextSegmentation: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.sigmoid.value,
                }
            },
        },
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.sigmoid.value,
                }
            },
        },
    },
    LayerOutputTypeChoice.Regression: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.linear.value,
                }
            },
        }
    },
    LayerOutputTypeChoice.Timeseries: {
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.linear.value,
                }
            },
        },
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "parameters": {
                "main": {
                    "activation": ActivationChoice.linear.value,
                }
            },
        },
    },
    LayerOutputTypeChoice.ObjectDetection: {
        "3D": {
            "type": LayerTypeChoice.Reshape.value,
            "parameters": {
                "main": {
                    "target_shape": "@shape",
                },
            },
        },
    },
}


KerasInstructions = {
    "mnist": {
        1: {
            "1_mnist": {
                "cols_names": "1_mnist",
                "height": 28,
                "width": 28,
                "max_scaler": 1,
                "min_scaler": 0,
                "net": "convolutional",
                "put": 1,
                "put_type": "image",
                "scaler": "min_max_scaler",
            }
        },
        2: {
            "2_classification": {
                "one_hot_encoding": True,
                "type_processing": "categorical",
            }
        },
    },
    "fashion_mnist": {
        1: {
            "1_fashion_mnist": {
                "cols_names": "1_fashion_mnist",
                "height": 28,
                "width": 28,
                "max_scaler": 1,
                "min_scaler": 0,
                "net": "convolutional",
                "put": 1,
                "put_type": "image",
                "scaler": "min_max_scaler",
            }
        },
        2: {
            "2_classification": {
                "one_hot_encoding": True,
                "type_processing": "categorical",
            }
        },
    },
    "cifar10": {
        1: {
            "1_cifar10": {
                "cols_names": "1_cifar10",
                "height": 32,
                "width": 32,
                "max_scaler": 1,
                "min_scaler": 0,
                "net": "convolutional",
                "put": 1,
                "put_type": "image",
                "scaler": "min_max_scaler",
            }
        },
        2: {
            "2_classification": {
                "one_hot_encoding": True,
                "type_processing": "categorical",
            }
        },
    },
    "cifar100": {
        1: {
            "1_cifar100": {
                "cols_names": "1_cifar100",
                "height": 32,
                "width": 32,
                "max_scaler": 1,
                "min_scaler": 0,
                "net": "convolutional",
                "put": 1,
                "put_type": "image",
                "scaler": "min_max_scaler",
            }
        },
        2: {
            "2_classification": {
                "one_hot_encoding": True,
                "type_processing": "categorical",
            }
        },
    },
}


DatasetCommonGroup = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "alias": "mnist",
                "name": "Mnist",
                "architecture": ArchitectureChoice.ImageClassification,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                        "inputs": {
                            1: {
                                "datatype": "2D",
                                "dtype": "float32",
                                "name": "Вход 1",
                                "shape": (28, 28, 1),
                                "task": LayerInputTypeChoice.Image,
                                "num_classes": 1,
                                "classes_names": ["mnist"],
                                "encoding": LayerEncodingChoice.none,
                            }
                        },
                        "outputs": {
                            2: {
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "name": "Выход 1",
                                "shape": (10,),
                                "task": LayerOutputTypeChoice.Classification,
                                "num_classes": 10,
                                "classes_names": [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                                "encoding": LayerEncodingChoice.ohe,
                            }
                        },
                        "columns": {
                            1: {
                                "1_mnist": {
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "name": "Вход 1",
                                    "shape": (28, 28, 1),
                                    "task": LayerInputTypeChoice.Image,
                                    "num_classes": 1,
                                    "classes_names": ["mnist"],
                                    "encoding": LayerEncodingChoice.none,
                                }
                            },
                            2: {
                                "2_classification": {
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "name": "Выход 1",
                                    "shape": (10,),
                                    "task": LayerOutputTypeChoice.Classification,
                                    "num_classes": 10,
                                    "classes_names": [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                    "encoding": LayerEncodingChoice.ohe,
                                }
                            },
                        },
                    },
                ],
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "architecture": ArchitectureChoice.ImageClassification,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                        "inputs": {
                            1: {
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": (28, 28, 1),
                                "name": "Вход 1",
                                "task": LayerInputTypeChoice.Image,
                                "classes_names": ["fashion_mnist"],
                                "num_classes": 1,
                            }
                        },
                        "outputs": {
                            2: {
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": (10,),
                                "name": "Выход 1",
                                "task": LayerOutputTypeChoice.Classification,
                                "classes_names": [
                                    "T-shirt/top",
                                    "Trouser",
                                    "Pullover",
                                    "Dress",
                                    "Coat",
                                    "Sandal",
                                    "Shirt",
                                    "Sneaker",
                                    "Bag",
                                    "Ankle boot",
                                ],
                                "num_classes": 10,
                                "encoding": "ohe",
                            }
                        },
                        "columns": {
                            1: {
                                "1_fashion_mnist": {
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "shape": (28, 28, 1),
                                    "name": "Вход 1",
                                    "task": LayerInputTypeChoice.Image,
                                    "classes_names": ["fashion_mnist"],
                                    "num_classes": 1,
                                }
                            },
                            2: {
                                "2_classification": {
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": (10,),
                                    "name": "Выход 1",
                                    "task": LayerOutputTypeChoice.Classification,
                                    "classes_names": [
                                        "T-shirt/top",
                                        "Trouser",
                                        "Pullover",
                                        "Dress",
                                        "Coat",
                                        "Sandal",
                                        "Shirt",
                                        "Sneaker",
                                        "Bag",
                                        "Ankle boot",
                                    ],
                                    "num_classes": 10,
                                    "encoding": "ohe",
                                }
                            },
                        },
                    },
                ],
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "architecture": ArchitectureChoice.ImageClassification,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                        "inputs": {
                            1: {
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": (32, 32, 3),
                                "name": "Вход 1",
                                "task": LayerInputTypeChoice.Image,
                                "classes_names": ["cifar10"],
                                "num_classes": 1,
                            }
                        },
                        "outputs": {
                            2: {
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": (10,),
                                "name": "Выход 1",
                                "task": LayerOutputTypeChoice.Classification,
                                "classes_names": [
                                    "airplane",
                                    "automobile",
                                    "bird",
                                    "cat",
                                    "deer",
                                    "dog",
                                    "frog",
                                    "horse",
                                    "ship",
                                    "truck",
                                ],
                                "num_classes": 10,
                                "encoding": "ohe",
                            }
                        },
                        "columns": {
                            1: {
                                "1_cifar10": {
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "shape": (32, 32, 3),
                                    "name": "Вход 1",
                                    "task": LayerInputTypeChoice.Image,
                                    "classes_names": ["cifar10"],
                                    "num_classes": 1,
                                }
                            },
                            2: {
                                "2_classification": {
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": (10,),
                                    "name": "Выход 1",
                                    "task": LayerOutputTypeChoice.Classification,
                                    "classes_names": [
                                        "airplane",
                                        "automobile",
                                        "bird",
                                        "cat",
                                        "deer",
                                        "dog",
                                        "frog",
                                        "horse",
                                        "ship",
                                        "truck",
                                    ],
                                    "num_classes": 10,
                                    "encoding": "ohe",
                                }
                            },
                        },
                    },
                ],
            },
            {
                "alias": "cifar100",
                "name": "Cifar 100",
                "architecture": ArchitectureChoice.ImageClassification,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                        "inputs": {
                            1: {
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": (32, 32, 3),
                                "name": "Вход 1",
                                "task": LayerInputTypeChoice.Image.value,
                                "classes_names": ["cifar100"],
                                "num_classes": 1,
                            },
                        },
                        "outputs": {
                            2: {
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": (100,),
                                "name": "Выход 1",
                                "task": LayerOutputTypeChoice.Classification.value,
                                "classes_names": [
                                    "apple",
                                    "aquarium_fish",
                                    "baby",
                                    "bear",
                                    "beaver",
                                    "bed",
                                    "bee",
                                    "beetle",
                                    "bicycle",
                                    "bottle",
                                    "bowl",
                                    "boy",
                                    "bridge",
                                    "bus",
                                    "butterfly",
                                    "camel",
                                    "can",
                                    "castle",
                                    "caterpillar",
                                    "cattle",
                                    "chair",
                                    "chimpanzee",
                                    "clock",
                                    "cloud",
                                    "cockroach",
                                    "couch",
                                    "cra",
                                    "crocodile",
                                    "cup",
                                    "dinosaur",
                                    "dolphin",
                                    "elephant",
                                    "flatfish",
                                    "forest",
                                    "fox",
                                    "girl",
                                    "hamster",
                                    "house",
                                    "kangaroo",
                                    "keyboard",
                                    "lamp",
                                    "lawn_mower",
                                    "leopard",
                                    "lion",
                                    "lizard",
                                    "lobster",
                                    "man",
                                    "maple_tree",
                                    "motorcycle",
                                    "mountain",
                                    "mouse",
                                    "mushroom",
                                    "oak_tree",
                                    "orange",
                                    "orchid",
                                    "otter",
                                    "palm_tree",
                                    "pear",
                                    "pickup_truck",
                                    "pine_tree",
                                    "plain",
                                    "plate",
                                    "poppy",
                                    "porcupine",
                                    "possum",
                                    "rabbit",
                                    "raccoon",
                                    "ray",
                                    "road",
                                    "rocket",
                                    "rose",
                                    "sea",
                                    "seal",
                                    "shark",
                                    "shrew",
                                    "skunk",
                                    "skyscraper",
                                    "snail",
                                    "snake",
                                    "spider",
                                    "squirrel",
                                    "streetcar",
                                    "sunflower",
                                    "sweet_pepper",
                                    "table",
                                    "tank",
                                    "telephone",
                                    "television",
                                    "tiger",
                                    "tractor",
                                    "train",
                                    "trout",
                                    "tulip",
                                    "turtle",
                                    "wardrobe",
                                    "whale",
                                    "willow_tree",
                                    "wolf",
                                    "woman",
                                    "worm",
                                ],
                                "num_classes": 100,
                                "encoding": "ohe",
                            }
                        },
                        "columns": {
                            1: {
                                "1_cifar100": {
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "shape": (32, 32, 3),
                                    "name": "Вход 1",
                                    "task": LayerInputTypeChoice.Image.value,
                                    "classes_names": ["cifar100"],
                                    "num_classes": 1,
                                }
                            },
                            2: {
                                "2_classification": {
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": (100,),
                                    "name": "Выход 1",
                                    "task": LayerOutputTypeChoice.Classification.value,
                                    "classes_names": [
                                        "apple",
                                        "aquarium_fish",
                                        "baby",
                                        "bear",
                                        "beaver",
                                        "bed",
                                        "bee",
                                        "beetle",
                                        "bicycle",
                                        "bottle",
                                        "bowl",
                                        "boy",
                                        "bridge",
                                        "bus",
                                        "butterfly",
                                        "camel",
                                        "can",
                                        "castle",
                                        "caterpillar",
                                        "cattle",
                                        "chair",
                                        "chimpanzee",
                                        "clock",
                                        "cloud",
                                        "cockroach",
                                        "couch",
                                        "cra",
                                        "crocodile",
                                        "cup",
                                        "dinosaur",
                                        "dolphin",
                                        "elephant",
                                        "flatfish",
                                        "forest",
                                        "fox",
                                        "girl",
                                        "hamster",
                                        "house",
                                        "kangaroo",
                                        "keyboard",
                                        "lamp",
                                        "lawn_mower",
                                        "leopard",
                                        "lion",
                                        "lizard",
                                        "lobster",
                                        "man",
                                        "maple_tree",
                                        "motorcycle",
                                        "mountain",
                                        "mouse",
                                        "mushroom",
                                        "oak_tree",
                                        "orange",
                                        "orchid",
                                        "otter",
                                        "palm_tree",
                                        "pear",
                                        "pickup_truck",
                                        "pine_tree",
                                        "plain",
                                        "plate",
                                        "poppy",
                                        "porcupine",
                                        "possum",
                                        "rabbit",
                                        "raccoon",
                                        "ray",
                                        "road",
                                        "rocket",
                                        "rose",
                                        "sea",
                                        "seal",
                                        "shark",
                                        "shrew",
                                        "skunk",
                                        "skyscraper",
                                        "snail",
                                        "snake",
                                        "spider",
                                        "squirrel",
                                        "streetcar",
                                        "sunflower",
                                        "sweet_pepper",
                                        "table",
                                        "tank",
                                        "telephone",
                                        "television",
                                        "tiger",
                                        "tractor",
                                        "train",
                                        "trout",
                                        "tulip",
                                        "turtle",
                                        "wardrobe",
                                        "whale",
                                        "willow_tree",
                                        "wolf",
                                        "woman",
                                        "worm",
                                    ],
                                    "num_classes": 100,
                                    "encoding": "ohe",
                                }
                            },
                        },
                    },
                ],
            },
        ],
    },
    {
        "alias": "terra",
        "name": "Terra",
        "datasets": [
            {
                "alias": "cars",
                "name": "Автомобили",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "cars_30",
                "name": "Автомобили (30 классов)",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "airplane",
                "name": "Самолеты",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "samoleti_drugaja",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "guby",
                "name": "Губы",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "ljudi",
                "name": "Люди",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "monety",
                "name": "Монеты",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "zapisi_s_videoregistratora",
                "name": "Записи с видеорегистратора",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "video_new",
                "name": "Видео",
                "tags": [
                    Tags.video,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "bus_passengers",
                "name": "Пассажиры автобусов",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "heads",
                "name": "Пассажиры автобусов (попарно)",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "marki_moloka",
                "name": "Марки молока",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "chess_v3",
                "name": "Шахматы v3",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "chess_v4_new",
                "name": "Шахматы v4 (генератор)",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "milk_v4_new",
                "name": "Молоко v4 (генератор)",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "cards_v4",
                "name": "Игральные карты v4",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "potholes_v4",
                "name": "Ямы на дорогах v4",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "trash_v4_new",
                "name": "Подводный мусор v4 (генератор)",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "bus_split_new",
                "name": "Автобусы v4 (генератор)",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "symptoms",
                "name": "Симптомы",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "writers",
                "name": "Тексты писателей",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "tesla",
                "name": "Отзывы на Теслу",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "symptoms_bow",
                "name": "Симптомы (bow)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "writers_bow",
                "name": "Тексты писателей (bow)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "tesla_bow",
                "name": "Отзывы на Теслу (bow)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "symptoms_w2v",
                "name": "Симптомы (w2v)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "writers_w2v",
                "name": "Тексты писателей (w2v)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "tesla_w2v",
                "name": "Отзывы на Теслу (w2v)",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "docs",
                "name": "Договоры",
                "tags": [
                    Tags.text,
                    Tags.text_segmentation,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "smarthome",
                "name": "Умный дом",
                "tags": [
                    Tags.audio,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "audio_genres",
                "name": "Жанры музыки",
                "tags": [
                    Tags.audio,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "esc_50",
                "name": "Окружающая среда",
                "tags": [
                    Tags.audio,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "kvartiry",
                "name": "Квартиры",
                "tags": [
                    Tags.classification,
                    Tags.text,
                    Tags.regression,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "rezjume",
                "name": "Резюме",
                "tags": [
                    Tags.classification,
                    Tags.text,
                    Tags.regression,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "tsena_avtomobilej",
                "name": "Цена автомобилей",
                "tags": [
                    Tags.classification,
                    Tags.regression,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "spam_soobschenija",
                "name": "Спам сообщения",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "klassifikatsija_rezjume",
                "name": "Классификация резюме",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "lukojl",
                "name": "Лукойл",
                "tags": [
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "lukoil_trend",
                "name": "Лукойл (тренд)",
                "tags": [
                    Tags.timeseriestrend,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "polimetall",
                "name": "Полиметалл",
                "tags": [
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "trafik_sajta",
                "name": "Трафик сайта",
                "tags": [
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "bus_video_tracker",
                "name": "Видео для трекера",
                "tags": [
                    Tags.video,
                    Tags.tracker,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "video_progulka_po_piteru",
                "name": "Видео_прогулка_по_Питеру",
                "tags": [
                    Tags.video,
                    Tags.tracker,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "audio_kniga_hobit_60sek",
                "name": "аудио_книга_Хобит_60сек",
                "tags": [
                    Tags.text,
                    Tags.speech_to_text,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "teksty_pisatelej_1000",
                "name": "тексты_писателей_1000",
                "tags": [
                    Tags.text,
                    Tags.text_to_speech,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "chess_tracker",
                "name": "Трекер шахматы",
                "tags": [
                    Tags.video,
                    Tags.tracker,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "brain_tumor",
                "name": "Опухоль мозга",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "breast_cancer",
                "name": "Рак груди",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "pnevmonija",
                "name": "Пневмония",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "obnaruzhenie_znachimyh_obektov_1",
                "name": "Обнаружение значимых объектов 1",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "obnaruzhenie_znachimyh_obektov_2",
                "name": "Обнаружение значимых объектов 2",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "maski_na_litse",
                "name": "Маски на лице",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "obnaruzhenie_ognja",
                "name": "Обнаружение огня",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "ufc",
                "name": "UFC",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "kaski_i_zhilety",
                "name": "Каски и жилеты",
                "tags": [
                    Tags.image,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "mnist_gan",
                "name": "Mnist GAN",
                "tags": [
                    Tags.image,
                    Tags.gan,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "chasy_gan",
                "name": "Часы GAN",
                "tags": [
                    Tags.image,
                    Tags.gan,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
            {
                "alias": "mnist_cgan",
                "name": "Mnist CGAN",
                "tags": [
                    Tags.image,
                    Tags.cgan,
                    Tags.terra_ai,
                ],
                "versions": [
                    {
                        "alias": "default",
                        "name": "Default",
                    },
                ],
            },
        ],
    },
    {
        "alias": "custom",
        "name": "Пользовательские",
        "datasets": [],
    },
]


VersionsGroups = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "mnist": [
                    {
                        "alias": "default",
                        "name": "Стандартная",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения цифр",
                                "datatype": "1D",
                                "dtype": "uint8",
                                "shape": [28, 28],
                                "num_classes": 10,
                                "classes_names": [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [10],
                                "num_classes": 10,
                                "classes_names": [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения цифр",
                                    "datatype": "1D",
                                    "dtype": "float32",
                                    "shape": [28, 28],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [10],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    },
                    {
                        "alias": "add_dimension",
                        "name": "Добавленная размерность",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения цифр",
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": [28, 28, 1],
                                "num_classes": 10,
                                "classes_names": [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [10],
                                "num_classes": 10,
                                "classes_names": [
                                    "0",
                                    "1",
                                    "2",
                                    "3",
                                    "4",
                                    "5",
                                    "6",
                                    "7",
                                    "8",
                                    "9",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения цифр",
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "shape": [28, 28, 1],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [10],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "0",
                                        "1",
                                        "2",
                                        "3",
                                        "4",
                                        "5",
                                        "6",
                                        "7",
                                        "8",
                                        "9",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    },
                ],
                "fashion_mnist": [
                    {
                        "alias": "default",
                        "name": "Стандартная",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения одежды",
                                "datatype": "1D",
                                "dtype": "uint8",
                                "shape": [28, 28],
                                "num_classes": 10,
                                "classes_names": [
                                    "T-shirt/top",
                                    "Trouser",
                                    "Pullover",
                                    "Dress",
                                    "Coat",
                                    "Sandal",
                                    "Shirt",
                                    "Sneaker",
                                    "Bag",
                                    "Ankle boot",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [10],
                                "num_classes": 10,
                                "classes_names": [
                                    "T-shirt/top",
                                    "Trouser",
                                    "Pullover",
                                    "Dress",
                                    "Coat",
                                    "Sandal",
                                    "Shirt",
                                    "Sneaker",
                                    "Bag",
                                    "Ankle boot",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения одежды",
                                    "datatype": "1D",
                                    "dtype": "float32",
                                    "shape": [28, 28],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "T-shirt/top",
                                        "Trouser",
                                        "Pullover",
                                        "Dress",
                                        "Coat",
                                        "Sandal",
                                        "Shirt",
                                        "Sneaker",
                                        "Bag",
                                        "Ankle boot",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [10],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "T-shirt/top",
                                        "Trouser",
                                        "Pullover",
                                        "Dress",
                                        "Coat",
                                        "Sandal",
                                        "Shirt",
                                        "Sneaker",
                                        "Bag",
                                        "Ankle boot",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    },
                    {
                        "alias": "add_dimension",
                        "name": "Добавленная размерность",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения цифр",
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": [28, 28, 1],
                                "num_classes": 10,
                                "classes_names": [
                                    "T-shirt/top",
                                    "Trouser",
                                    "Pullover",
                                    "Dress",
                                    "Coat",
                                    "Sandal",
                                    "Shirt",
                                    "Sneaker",
                                    "Bag",
                                    "Ankle boot",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [10],
                                "num_classes": 10,
                                "classes_names": [
                                    "T-shirt/top",
                                    "Trouser",
                                    "Pullover",
                                    "Dress",
                                    "Coat",
                                    "Sandal",
                                    "Shirt",
                                    "Sneaker",
                                    "Bag",
                                    "Ankle boot",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения цифр",
                                    "datatype": "2D",
                                    "dtype": "float32",
                                    "shape": [28, 28, 1],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "T-shirt/top",
                                        "Trouser",
                                        "Pullover",
                                        "Dress",
                                        "Coat",
                                        "Sandal",
                                        "Shirt",
                                        "Sneaker",
                                        "Bag",
                                        "Ankle boot",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [10],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "T-shirt/top",
                                        "Trouser",
                                        "Pullover",
                                        "Dress",
                                        "Coat",
                                        "Sandal",
                                        "Shirt",
                                        "Sneaker",
                                        "Bag",
                                        "Ankle boot",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    },
                ],
                "cifar10": [
                    {
                        "alias": "default",
                        "name": "Стандартная",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения",
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": [32, 32, 3],
                                "num_classes": 10,
                                "classes_names": [
                                    "airplane",
                                    "automobile",
                                    "bird",
                                    "cat",
                                    "deer",
                                    "dog",
                                    "frog",
                                    "horse",
                                    "ship",
                                    "truck",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [10],
                                "num_classes": 10,
                                "classes_names": [
                                    "airplane",
                                    "automobile",
                                    "bird",
                                    "cat",
                                    "deer",
                                    "dog",
                                    "frog",
                                    "horse",
                                    "ship",
                                    "truck",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения одежды",
                                    "datatype": "1D",
                                    "dtype": "float32",
                                    "shape": [32, 32, 3],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "airplane",
                                        "automobile",
                                        "bird",
                                        "cat",
                                        "deer",
                                        "dog",
                                        "frog",
                                        "horse",
                                        "ship",
                                        "truck",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [10],
                                    "num_classes": 10,
                                    "classes_names": [
                                        "airplane",
                                        "automobile",
                                        "bird",
                                        "cat",
                                        "deer",
                                        "dog",
                                        "frog",
                                        "horse",
                                        "ship",
                                        "truck",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    }
                ],
                "cifar100": [
                    {
                        "alias": "default",
                        "name": "Стандартная",
                        "date": datetime.now().isoformat(),
                        "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                        "use_generator": False,
                        "inputs": {
                            "1": {
                                "name": "Изображения",
                                "datatype": "2D",
                                "dtype": "float32",
                                "shape": [32, 32, 3],
                                "num_classes": 100,
                                "classes_names": [
                                    "apple",
                                    "aquarium_fish",
                                    "baby",
                                    "bear",
                                    "beaver",
                                    "bed",
                                    "bee",
                                    "beetle",
                                    "bicycle",
                                    "bottle",
                                    "bowl",
                                    "boy",
                                    "bridge",
                                    "bus",
                                    "butterfly",
                                    "camel",
                                    "can",
                                    "castle",
                                    "caterpillar",
                                    "cattle",
                                    "chair",
                                    "chimpanzee",
                                    "clock",
                                    "cloud",
                                    "cockroach",
                                    "couch",
                                    "cra",
                                    "crocodile",
                                    "cup",
                                    "dinosaur",
                                    "dolphin",
                                    "elephant",
                                    "flatfish",
                                    "forest",
                                    "fox",
                                    "girl",
                                    "hamster",
                                    "house",
                                    "kangaroo",
                                    "keyboard",
                                    "lamp",
                                    "lawn_mower",
                                    "leopard",
                                    "lion",
                                    "lizard",
                                    "lobster",
                                    "man",
                                    "maple_tree",
                                    "motorcycle",
                                    "mountain",
                                    "mouse",
                                    "mushroom",
                                    "oak_tree",
                                    "orange",
                                    "orchid",
                                    "otter",
                                    "palm_tree",
                                    "pear",
                                    "pickup_truck",
                                    "pine_tree",
                                    "plain",
                                    "plate",
                                    "poppy",
                                    "porcupine",
                                    "possum",
                                    "rabbit",
                                    "raccoon",
                                    "ray",
                                    "road",
                                    "rocket",
                                    "rose",
                                    "sea",
                                    "seal",
                                    "shark",
                                    "shrew",
                                    "skunk",
                                    "skyscraper",
                                    "snail",
                                    "snake",
                                    "spider",
                                    "squirrel",
                                    "streetcar",
                                    "sunflower",
                                    "sweet_pepper",
                                    "table",
                                    "tank",
                                    "telephone",
                                    "television",
                                    "tiger",
                                    "tractor",
                                    "train",
                                    "trout",
                                    "tulip",
                                    "turtle",
                                    "wardrobe",
                                    "whale",
                                    "willow_tree",
                                    "wolf",
                                    "woman",
                                    "worm",
                                ],
                                "classes_colors": None,
                                "encoding": "none",
                                "task": "Image",
                            }
                        },
                        "outputs": {
                            "2": {
                                "name": "Метки классов",
                                "datatype": "DIM",
                                "dtype": "uint8",
                                "shape": [100],
                                "num_classes": 100,
                                "classes_names": [
                                    "apple",
                                    "aquarium_fish",
                                    "baby",
                                    "bear",
                                    "beaver",
                                    "bed",
                                    "bee",
                                    "beetle",
                                    "bicycle",
                                    "bottle",
                                    "bowl",
                                    "boy",
                                    "bridge",
                                    "bus",
                                    "butterfly",
                                    "camel",
                                    "can",
                                    "castle",
                                    "caterpillar",
                                    "cattle",
                                    "chair",
                                    "chimpanzee",
                                    "clock",
                                    "cloud",
                                    "cockroach",
                                    "couch",
                                    "cra",
                                    "crocodile",
                                    "cup",
                                    "dinosaur",
                                    "dolphin",
                                    "elephant",
                                    "flatfish",
                                    "forest",
                                    "fox",
                                    "girl",
                                    "hamster",
                                    "house",
                                    "kangaroo",
                                    "keyboard",
                                    "lamp",
                                    "lawn_mower",
                                    "leopard",
                                    "lion",
                                    "lizard",
                                    "lobster",
                                    "man",
                                    "maple_tree",
                                    "motorcycle",
                                    "mountain",
                                    "mouse",
                                    "mushroom",
                                    "oak_tree",
                                    "orange",
                                    "orchid",
                                    "otter",
                                    "palm_tree",
                                    "pear",
                                    "pickup_truck",
                                    "pine_tree",
                                    "plain",
                                    "plate",
                                    "poppy",
                                    "porcupine",
                                    "possum",
                                    "rabbit",
                                    "raccoon",
                                    "ray",
                                    "road",
                                    "rocket",
                                    "rose",
                                    "sea",
                                    "seal",
                                    "shark",
                                    "shrew",
                                    "skunk",
                                    "skyscraper",
                                    "snail",
                                    "snake",
                                    "spider",
                                    "squirrel",
                                    "streetcar",
                                    "sunflower",
                                    "sweet_pepper",
                                    "table",
                                    "tank",
                                    "telephone",
                                    "television",
                                    "tiger",
                                    "tractor",
                                    "train",
                                    "trout",
                                    "tulip",
                                    "turtle",
                                    "wardrobe",
                                    "whale",
                                    "willow_tree",
                                    "wolf",
                                    "woman",
                                    "worm",
                                ],
                                "classes_colors": None,
                                "encoding": "ohe",
                                "task": "Classification",
                            }
                        },
                        "service": {},
                        "columns": {
                            "1": {
                                "1_image": {
                                    "name": "Изображения одежды",
                                    "datatype": "1D",
                                    "dtype": "float32",
                                    "shape": [32, 32, 3],
                                    "num_classes": 100,
                                    "classes_names": [
                                        "apple",
                                        "aquarium_fish",
                                        "baby",
                                        "bear",
                                        "beaver",
                                        "bed",
                                        "bee",
                                        "beetle",
                                        "bicycle",
                                        "bottle",
                                        "bowl",
                                        "boy",
                                        "bridge",
                                        "bus",
                                        "butterfly",
                                        "camel",
                                        "can",
                                        "castle",
                                        "caterpillar",
                                        "cattle",
                                        "chair",
                                        "chimpanzee",
                                        "clock",
                                        "cloud",
                                        "cockroach",
                                        "couch",
                                        "cra",
                                        "crocodile",
                                        "cup",
                                        "dinosaur",
                                        "dolphin",
                                        "elephant",
                                        "flatfish",
                                        "forest",
                                        "fox",
                                        "girl",
                                        "hamster",
                                        "house",
                                        "kangaroo",
                                        "keyboard",
                                        "lamp",
                                        "lawn_mower",
                                        "leopard",
                                        "lion",
                                        "lizard",
                                        "lobster",
                                        "man",
                                        "maple_tree",
                                        "motorcycle",
                                        "mountain",
                                        "mouse",
                                        "mushroom",
                                        "oak_tree",
                                        "orange",
                                        "orchid",
                                        "otter",
                                        "palm_tree",
                                        "pear",
                                        "pickup_truck",
                                        "pine_tree",
                                        "plain",
                                        "plate",
                                        "poppy",
                                        "porcupine",
                                        "possum",
                                        "rabbit",
                                        "raccoon",
                                        "ray",
                                        "road",
                                        "rocket",
                                        "rose",
                                        "sea",
                                        "seal",
                                        "shark",
                                        "shrew",
                                        "skunk",
                                        "skyscraper",
                                        "snail",
                                        "snake",
                                        "spider",
                                        "squirrel",
                                        "streetcar",
                                        "sunflower",
                                        "sweet_pepper",
                                        "table",
                                        "tank",
                                        "telephone",
                                        "television",
                                        "tiger",
                                        "tractor",
                                        "train",
                                        "trout",
                                        "tulip",
                                        "turtle",
                                        "wardrobe",
                                        "whale",
                                        "willow_tree",
                                        "wolf",
                                        "woman",
                                        "worm",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "none",
                                    "task": "Image",
                                }
                            },
                            "2": {
                                "2_classification": {
                                    "name": "Метки классов",
                                    "datatype": "DIM",
                                    "dtype": "uint8",
                                    "shape": [100],
                                    "num_classes": 100,
                                    "classes_names": [
                                        "apple",
                                        "aquarium_fish",
                                        "baby",
                                        "bear",
                                        "beaver",
                                        "bed",
                                        "bee",
                                        "beetle",
                                        "bicycle",
                                        "bottle",
                                        "bowl",
                                        "boy",
                                        "bridge",
                                        "bus",
                                        "butterfly",
                                        "camel",
                                        "can",
                                        "castle",
                                        "caterpillar",
                                        "cattle",
                                        "chair",
                                        "chimpanzee",
                                        "clock",
                                        "cloud",
                                        "cockroach",
                                        "couch",
                                        "cra",
                                        "crocodile",
                                        "cup",
                                        "dinosaur",
                                        "dolphin",
                                        "elephant",
                                        "flatfish",
                                        "forest",
                                        "fox",
                                        "girl",
                                        "hamster",
                                        "house",
                                        "kangaroo",
                                        "keyboard",
                                        "lamp",
                                        "lawn_mower",
                                        "leopard",
                                        "lion",
                                        "lizard",
                                        "lobster",
                                        "man",
                                        "maple_tree",
                                        "motorcycle",
                                        "mountain",
                                        "mouse",
                                        "mushroom",
                                        "oak_tree",
                                        "orange",
                                        "orchid",
                                        "otter",
                                        "palm_tree",
                                        "pear",
                                        "pickup_truck",
                                        "pine_tree",
                                        "plain",
                                        "plate",
                                        "poppy",
                                        "porcupine",
                                        "possum",
                                        "rabbit",
                                        "raccoon",
                                        "ray",
                                        "road",
                                        "rocket",
                                        "rose",
                                        "sea",
                                        "seal",
                                        "shark",
                                        "shrew",
                                        "skunk",
                                        "skyscraper",
                                        "snail",
                                        "snake",
                                        "spider",
                                        "squirrel",
                                        "streetcar",
                                        "sunflower",
                                        "sweet_pepper",
                                        "table",
                                        "tank",
                                        "telephone",
                                        "television",
                                        "tiger",
                                        "tractor",
                                        "train",
                                        "trout",
                                        "tulip",
                                        "turtle",
                                        "wardrobe",
                                        "whale",
                                        "willow_tree",
                                        "wolf",
                                        "woman",
                                        "worm",
                                    ],
                                    "classes_colors": None,
                                    "encoding": "ohe",
                                    "task": "Classification",
                                }
                            },
                        },
                    }
                ],
            }
        ],
    }
]

DatasetsGroups = []
