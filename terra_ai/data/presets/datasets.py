"""
Предустановки датасетов
"""

from enum import Enum
from terra_ai.data.datasets.extra import (
    DatasetGroupChoice,
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerEncodingChoice,
)
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.modeling.layers.extra import ActivationChoice


class Tags(dict, Enum):
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


OutputLayersDefaults = {
    LayerOutputTypeChoice.Classification: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "activation": ActivationChoice.softmax.value,
        },
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "activation": ActivationChoice.softmax.value,
        },
    },
    LayerOutputTypeChoice.Segmentation: {
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "activation": ActivationChoice.softmax.value,
        },
        "2D": {
            "type": LayerTypeChoice.Conv2D.value,
            "activation": ActivationChoice.softmax.value,
        },
        "3D": {
            "type": LayerTypeChoice.Conv3D.value,
            "activation": ActivationChoice.softmax.value,
        },
    },
    LayerOutputTypeChoice.TextSegmentation: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "activation": ActivationChoice.sigmoid.value,
        },
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "activation": ActivationChoice.sigmoid.value,
        },
    },
    LayerOutputTypeChoice.Regression: {
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "activation": ActivationChoice.linear.value,
        }
    },
    LayerOutputTypeChoice.Timeseries: {
        "1D": {
            "type": LayerTypeChoice.Conv1D.value,
            "activation": ActivationChoice.linear.value,
        },
        "DIM": {
            "type": LayerTypeChoice.Dense.value,
            "activation": ActivationChoice.linear.value,
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


DatasetsGroups = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "alias": "mnist",
                "name": "Mnist",
                "group": "keras",
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Вход 1",
                        "shape": (28, 28, 1),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["mnist"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Выход 1",
                        "shape": (10,),
                        "task": LayerOutputTypeChoice.Classification.value,
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
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "columns": {
                    1: {
                        "1_mnist": {
                            "datatype": "2D",
                            "dtype": "float32",
                            "name": "Вход 1",
                            "shape": (28, 28, 1),
                            "task": LayerInputTypeChoice.Image.value,
                            "num_classes": 1,
                            "classes_names": ["mnist"],
                            "encoding": LayerEncodingChoice.none.value,
                        }
                    },
                    2: {
                        "2_classification": {
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "name": "Выход 1",
                            "shape": (10,),
                            "task": LayerOutputTypeChoice.Classification.value,
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
                            "encoding": LayerEncodingChoice.ohe.value,
                        }
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
                "use_generator": False,
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "group": DatasetGroupChoice.keras.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (28, 28, 1),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Image.value,
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
                        "task": LayerOutputTypeChoice.Classification.value,
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
                            "task": LayerInputTypeChoice.Image.value,
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
                            "task": LayerOutputTypeChoice.Classification.value,
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
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
                "use_generator": False,
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "group": DatasetGroupChoice.keras.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (32, 32, 3),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Image.value,
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
                        "task": LayerOutputTypeChoice.Classification.value,
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
                            "task": LayerInputTypeChoice.Image.value,
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
                            "task": LayerOutputTypeChoice.Classification.value,
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
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
                "use_generator": False,
            },
            {
                "alias": "cifar100",
                "name": "Сifar 100",
                "group": DatasetGroupChoice.keras.value,
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
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
                "use_generator": False,
            },
        ],
    },
    {
        "alias": "terra",
        "name": "Terra",
        "datasets": [
            # {
            #     "alias": "sberbank_timeseries",
            #     "name": "Акции сбербанка",
            #     "group": DatasetGroupChoice.terra.value,
            #     "tags": [
            #         Tags.timeseries.value,
            #         Tags.terra_ai.value,
            #     ],
            # },
            {
                "alias": "cars",
                "name": "Автомобили",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (120, 176, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 3,
                        "classes_names": ["Мерседес", "Рено", "Феррари"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (3,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 3,
                        "classes_names": ["Мерседес", "Рено", "Феррари"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "cars_30",
                "name": "Автомобили (30 классов)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (48, 96, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 30,
                        "classes_names": ["Audi", "BMW", "Bentley", "Cadillac", "Chevrolet",
                                          "Citroen", "Dodge", "Fiat", "Ford", "GMC",
                                          "Honda", "Infiniti", "Jaguar", "Jeep", "KIA",
                                          "Land_Rover", "Lexus", "Mazda", "Mercedes_Benz", "Nissan",
                                          "Opel", "Peugeot", "Porsche", "Renault", "Rolls_Royce",
                                          "Skoda", "Subaru", "Toyota", "Volkswagen", "Volvo"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (30,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 3,
                        "classes_names": ["Audi", "BMW", "Bentley", "Cadillac", "Chevrolet",
                                          "Citroen", "Dodge", "Fiat", "Ford", "GMC",
                                          "Honda", "Infiniti", "Jaguar", "Jeep", "KIA",
                                          "Land_Rover", "Lexus", "Mazda", "Mercedes_Benz", "Nissan",
                                          "Opel", "Peugeot", "Porsche", "Renault", "Rolls_Royce",
                                          "Skoda", "Subaru", "Toyota", "Volkswagen", "Volvo"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "airplane",
                "name": "Самолеты",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (128, 160, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Самолеты"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (128, 160, 2),
                        "task": LayerOutputTypeChoice.Segmentation.value,
                        "num_classes": 2,
                        "classes_names": ["Небо", "Самолет"],
                        "classes_colors": ["black", "red"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.segmentation.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "guby",
                "name": "Губы",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (128, 160, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Губы"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (128, 160, 2),
                        "task": LayerOutputTypeChoice.Segmentation.value,
                        "num_classes": 2,
                        "classes_names": ["Окружение", "Губы"],
                        "classes_colors": ["black", "lime"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.segmentation.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "ljudi",
                "name": "Люди",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (160, 128, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Человек"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (160, 128, 2),
                        "task": LayerOutputTypeChoice.Segmentation.value,
                        "num_classes": 2,
                        "classes_names": ["Окружение", "Человек"],
                        "classes_colors": ["black", "white"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.segmentation.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "monety",
                "name": "Монеты",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (160, 160, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Photo"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (160, 160, 2),
                        "task": LayerOutputTypeChoice.Segmentation.value,
                        "num_classes": 2,
                        "classes_names": ["Фон", "Монета"],
                        "classes_colors": ["black", "white"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.segmentation.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "zapisi_s_videoregistratora",
                "name": "Записи с видеорегистратора",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (128, 160, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Дорога"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (128, 160, 3),
                        "task": LayerOutputTypeChoice.Segmentation.value,
                        "num_classes": 2,
                        "classes_names": ["Окружение", "Дорога", "Граница"],
                        "classes_colors": ["black", "#00dc6e", "white"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.segmentation.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "video",
                "name": "Видео",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "3D",
                        "dtype": "float64",
                        "name": "Input 1",
                        "shape": (64, 100, 120, 3),
                        "task": LayerInputTypeChoice.Video.value,
                        "num_classes": 5,
                        "classes_names": ["Cricket Shot", "Playing Cello", "Punch", "Shaving Beard", "Tennis Swing"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (5,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 5,
                        "classes_names": ["Cricket Shot", "Playing Cello", "Punch", "Shaving Beard", "Tennis Swing"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.video.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "bus_passengers",
                "name": "Пассажиры автобусов",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (200, 100, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 2,
                        "classes_names": ["Входящий", "Выходящий"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (2,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 2,
                        "classes_names": ["Входящий", "Выходящий"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "heads",
                "name": "Пассажиры автобусов (попарно)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (100, 100, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    2: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 2",
                        "shape": (100, 100, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "classes_names": ["Входящий", "Выходящий"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    3: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (2,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 2,
                        "classes_names": ["Входящий", "Выходящий"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "marki_moloka",
                "name": "Марки молока",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (200, 100, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 3,
                        "classes_names": ["Parmalat", "Кубанская буренка", "Семейный формат"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Output 1",
                        "shape": (3,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 3,
                        "classes_names": ["Parmalat", "Кубанская буренка", "Семейный формат"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "chess_v3",
                "name": "Шахматы v3",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Шахматы"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "chess_v4",
                "name": "Шахматы v4 (генератор)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Шахматы"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["black-knight", "white-rook", "white-pawn", "white-king", "white-bishop",
                                          "black-rook", "black-pawn", "black-king", "black-bishop", "black-queen",
                                          "white-queen", "white-knight", "bishop"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "milk_v4",
                "name": "Молоко v4 (генератор)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Images"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 10,
                        "classes_names": ["Beloe Ozero", "Domashkino", "Domik v Derevne", "Letnii-lug",
                                          "Prostokvashino", "Selo-zelenoe", "Stanciya-otbornoe", "Stanciya",
                                          "Tashlinki-p", "Tashlinki-u"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 10,
                        "classes_names": ["Beloe Ozero", "Domashkino", "Domik v Derevne", "Letnii-lug",
                                          "Prostokvashino", "Selo-zelenoe", "Stanciya-otbornoe", "Stanciya",
                                          "Tashlinki-p", "Tashlinki-u"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 10,
                        "classes_names": ["Beloe Ozero", "Domashkino", "Domik v Derevne", "Letnii-lug",
                                          "Prostokvashino", "Selo-zelenoe", "Stanciya-otbornoe", "Stanciya",
                                          "Tashlinki-p", "Tashlinki-u"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "cards_v4",
                "name": "Игральные карты v4",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Images"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 6,
                        "classes_names": ["ace", "jack", "king", "nine", "queen", "ten"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 6,
                        "classes_names": ["ace", "jack", "king", "nine", "queen", "ten"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 6,
                        "classes_names": ["ace", "jack", "king", "nine", "queen", "ten"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "potholes_v4",
                "name": "Ямы на дорогах v4",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Images"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["pothole"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["pothole"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["pothole"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "trash_v4",
                "name": "Подводный мусор v4 (генератор)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Images"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["bio", "cloth", "fishing", "metal", "paper", "papper", "plastic",
                                          "platstic", "rov", "rubber", "timestamp", "unknown", "wood"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["bio", "cloth", "fishing", "metal", "paper", "papper", "plastic",
                                          "platstic", "rov", "rubber", "timestamp", "unknown", "wood"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 13,
                        "classes_names": ["bio", "cloth", "fishing", "metal", "paper", "papper", "plastic",
                                          "platstic", "rov", "rubber", "timestamp", "unknown", "wood"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "bus",
                "name": "Автобусы v4 (генератор)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (416, 416, 3),
                        "task": LayerInputTypeChoice.Image.value,
                        "num_classes": 1,
                        "classes_names": ["Images"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (52, 52, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["person"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (26, 26, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["person"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    4: {
                        "datatype": "3D",
                        "dtype": "float32",
                        "name": "Bounding boxes",
                        "shape": (13, 13, 3, 18),
                        "task": LayerOutputTypeChoice.ObjectDetection.value,
                        "num_classes": 1,
                        "classes_names": ["person"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                },
                "tags": [
                    Tags.image.value,
                    Tags.object_detection.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "symptoms",
                "name": "Симптомы",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "name": "Input 1",
                        "shape": (100, ),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (10,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "writers",
                "name": "Тексты писателей",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "name": "Input 1",
                        "shape": (1000,),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри","Стругацкие"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (6,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри","Стругацкие"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "tesla",
                "name": "Отзывы на Теслу",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "name": "Input 1",
                        "shape": (100,),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (2,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "symptoms_bow",
                "name": "Симптомы (bow)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "name": "Input 1",
                        "shape": (1200,),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (10,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "writers_bow",
                "name": "Тексты писателей (bow)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "name": "Input 1",
                        "shape": (20000,),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри", "Стругацкие"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (6,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри", "Стругацкие"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "tesla_bow",
                "name": "Отзывы на Теслу (bow)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "name": "Input 1",
                        "shape": (20000,),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (2,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "symptoms_w2v",
                "name": "Симптомы (w2v)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (100, 200),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (10,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 10,
                        "classes_names": ["Аппендицит", "Гастрит", "Гепатит", "Дуоденит", "Колит",
                                          "Панкреатит", "Холецистит", "Эзофагит", "Энтерит", "Язва"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "writers_w2v",
                "name": "Тексты писателей (w2v)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (1000, 200),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри", "Стругацкие"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (6,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 6,
                        "classes_names": ["Булгаков", "Клиффорд Саймак", "Макс Фрай",
                                          "О. Генри", "Рэй Брэдберри", "Стругацкие"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "tesla_w2v",
                "name": "Отзывы на Теслу (w2v)",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Input 1",
                        "shape": (100, 200),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (2,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 2,
                        "classes_names": ["Положительные", "Отрицательные"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "docs",
                "name": "Договоры",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "name": "Input 1",
                        "shape": (80, ),
                        "task": LayerInputTypeChoice.Text.value,
                        "num_classes": 1,
                        "classes_names": ["Договоры"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    2: {
                        "datatype": "1D",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (80, 6),
                        "task": LayerOutputTypeChoice.TextSegmentation.value,
                        "num_classes": 6,
                        "classes_names": ["<s1>", "<s2>", "<s3>", "<s4>", "<s5>", "<s6>"],
                        "encoding": LayerEncodingChoice.multi.value,
                    }
                },
                "tags": [
                    Tags.text.value,
                    Tags.text_segmentation.value,
                    Tags.russian.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "smarthome",
                "name": "Умный дом",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "MFCC",
                        "shape": (44, 20),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 4,
                        "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    2: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Chroma STFT",
                        "shape": (44, 12),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 4,
                        "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "DIM",
                        "dtype": "float32",
                        "name": "RMS",
                        "shape": (44, ),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 4,
                        "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    4: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (4,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 4,
                        "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.audio.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "audio_genres",
                "name": "Жанры музыки",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "MFCC",
                        "shape": (1292, 20),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 10,
                        "classes_names": ["Блюз", "Джаз", "Диско", "Кантри", "Классика",
                                          "Металл", "Поп", "Регги", "Рок", "ХипХоп"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    2: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Chroma STFT",
                        "shape": (1292, 12),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 10,
                        "classes_names": ["Блюз", "Джаз", "Диско", "Кантри", "Классика",
                                          "Металл", "Поп", "Регги", "Рок", "ХипХоп"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "DIM",
                        "dtype": "float32",
                        "name": "RMS",
                        "shape": (1292,),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 10,
                        "classes_names": ["Блюз", "Джаз", "Диско", "Кантри", "Классика",
                                          "Металл", "Поп", "Регги", "Рок", "ХипХоп"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    4: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (10,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 10,
                        "classes_names": ["Блюз", "Джаз", "Диско", "Кантри", "Классика",
                                          "Металл", "Поп", "Регги", "Рок", "ХипХоп"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.audio.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "esc_50",
                "name": "Окружающая среда",
                "group": DatasetGroupChoice.terra.value,
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "MFCC",
                        "shape": (130, 20),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 50,
                        "classes_names": ["airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat",
                                          "chainsaw", "chirping_birds", "church_bells", "clapping", "clock_alarm",
                                          "clock_tick", "coughing", "cow", "crackling_fire", "crickets", "crow",
                                          "crying_baby", "dog", "door_wood_creaks", "door_wood_knock",
                                          "drinking_sipping", "engine", "fireworks", "footsteps", "frog",
                                          "glass_breaking", "hand_saw", "helicopter", "hen", "insects",
                                          "keyboard_typing", "laughing", "mouse_click", "pig", "pouring_water", "rain",
                                          "rooster", "sea_waves", "sheep", "siren", "sneezing", "snoring",
                                          "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine",
                                          "water_drops", "wind"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    2: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "name": "Chroma STFT",
                        "shape": (130, 12),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 50,
                        "classes_names": ["airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat",
                                          "chainsaw", "chirping_birds", "church_bells", "clapping", "clock_alarm",
                                          "clock_tick", "coughing", "cow", "crackling_fire", "crickets", "crow",
                                          "crying_baby", "dog", "door_wood_creaks", "door_wood_knock",
                                          "drinking_sipping", "engine", "fireworks", "footsteps", "frog",
                                          "glass_breaking", "hand_saw", "helicopter", "hen", "insects",
                                          "keyboard_typing", "laughing", "mouse_click", "pig", "pouring_water", "rain",
                                          "rooster", "sea_waves", "sheep", "siren", "sneezing", "snoring",
                                          "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine",
                                          "water_drops", "wind"],
                        "encoding": LayerEncodingChoice.none.value,
                    },
                    3: {
                        "datatype": "DIM",
                        "dtype": "float32",
                        "name": "RMS",
                        "shape": (130,),
                        "task": LayerInputTypeChoice.Audio.value,
                        "num_classes": 50,
                        "classes_names": ["airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat",
                                          "chainsaw", "chirping_birds", "church_bells", "clapping", "clock_alarm",
                                          "clock_tick", "coughing", "cow", "crackling_fire", "crickets", "crow",
                                          "crying_baby", "dog", "door_wood_creaks", "door_wood_knock",
                                          "drinking_sipping", "engine", "fireworks", "footsteps", "frog",
                                          "glass_breaking", "hand_saw", "helicopter", "hen", "insects",
                                          "keyboard_typing", "laughing", "mouse_click", "pig", "pouring_water", "rain",
                                          "rooster", "sea_waves", "sheep", "siren", "sneezing", "snoring",
                                          "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine",
                                          "water_drops", "wind"],
                        "encoding": LayerEncodingChoice.none.value,
                    }
                },
                "outputs": {
                    4: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "name": "Метки классов",
                        "shape": (50,),
                        "task": LayerOutputTypeChoice.Classification.value,
                        "num_classes": 50,
                        "classes_names": ["airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat",
                                          "chainsaw", "chirping_birds", "church_bells", "clapping", "clock_alarm",
                                          "clock_tick", "coughing", "cow", "crackling_fire", "crickets", "crow",
                                          "crying_baby", "dog", "door_wood_creaks", "door_wood_knock",
                                          "drinking_sipping", "engine", "fireworks", "footsteps", "frog",
                                          "glass_breaking", "hand_saw", "helicopter", "hen", "insects",
                                          "keyboard_typing", "laughing", "mouse_click", "pig", "pouring_water", "rain",
                                          "rooster", "sea_waves", "sheep", "siren", "sneezing", "snoring",
                                          "thunderstorm", "toilet_flush", "train", "vacuum_cleaner", "washing_machine",
                                          "water_drops", "wind"],
                        "encoding": LayerEncodingChoice.ohe.value,
                    }
                },
                "tags": [
                    Tags.audio.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "kvartiry",
                "name": "Квартиры",
                "group": DatasetGroupChoice.terra.value,

                "tags": [
                    Tags.classification.value,
                    Tags.text.value,
                    Tags.regression.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "rezjume",
                "name": "Резюме",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.classification.value,
                    Tags.text.value,
                    Tags.regression.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "tsena_avtomobilej",
                "name": "Цена автомобилей",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.classification.value,
                    Tags.regression.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "spam_soobschenija",
                "name": "Спам сообщения",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "klassifikatsija_rezjume",
                "name": "Классификация резюме",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.text.value,
                    Tags.classification.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "lukojl",
                "name": "Лукойл",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.timeseries.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "lukoil_trend",
                "name": "Лукойл (тренд)",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.timeseriestrend.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "polimetall",
                "name": "Полиметалл",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.timeseries.value,
                    Tags.terra_ai.value,
                ],
            },
            {
                "alias": "trafik_sajta",
                "name": "Трафик сайта",
                "group": DatasetGroupChoice.terra.value,
                "tags": [
                    Tags.timeseries.value,
                    Tags.terra_ai.value,
                ],
            },
        ],
    },
    {
        "alias": "custom",
        "name": "Собственные",
    },
]
