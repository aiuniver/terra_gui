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

# from terra_ai.data.training.extra import TaskChoice


class Tags(dict, Enum):
    image = {"alias": "image", "name": "Image"}
    classification = {"alias": "classification", "name": "Classification"}
    tensorflow_keras = {"alias": "tensorflow_keras", "name": "Tensorflow.keras"}
    text = {"alias": "text", "name": "Text"}
    english = {"alias": "english", "name": "English"}
    russian = {"alias": "russian", "name": "Russian"}
    regression = {"alias": "regression", "name": "Regression"}
    timeseries = {"alias": "timeseries", "name": "Timeseries"}
    terra_ai = {"alias": "terra_ai", "name": "Terra AI"}
    object_detection = {"alias": "object_detection", "name": "Object detection"}
    segmentation = {"alias": "segmentation", "name": "Segmentation"}
    text_segmentation = {"alias": "text_segmentation", "name": "Text Segmentation"}
    audio = {"alias": "audio", "name": "Audio"}
    smart_home = {"alias": "smart_home", "name": "Smart home"}
    trading = {"alias": "trading", "name": "Trading"}


OutputLayersDefaults = {
    LayerOutputTypeChoice.Classification: {
        "DIM": {"type": LayerTypeChoice.Dense, "activation": ActivationChoice.softmax},
        "1D": {"type": LayerTypeChoice.Conv1D, "activation": ActivationChoice.softmax},
    },
    LayerOutputTypeChoice.Segmentation: {
        "1D": {"type": LayerTypeChoice.Conv1D, "activation": ActivationChoice.softmax},
        "2D": {"type": LayerTypeChoice.Conv2D, "activation": ActivationChoice.softmax},
        "3D": {"type": LayerTypeChoice.Conv3D, "activation": ActivationChoice.softmax},
    },
    LayerOutputTypeChoice.TextSegmentation: {
        "DIM": {"type": LayerTypeChoice.Dense, "activation": ActivationChoice.sigmoid},
        "1D": {"type": LayerTypeChoice.Conv1D, "activation": ActivationChoice.sigmoid},
    },
    LayerOutputTypeChoice.Regression: {
        "DIM": {"type": LayerTypeChoice.Dense, "activation": ActivationChoice.linear}
    },
    LayerOutputTypeChoice.Timeseries: {
        "1D": {"type": LayerTypeChoice.Conv1D, "activation": ActivationChoice.linear},
        "DIM": {"type": LayerTypeChoice.Dense, "activation": ActivationChoice.linear},
    },
}

KerasInstructions = {
    "mnist": {
        1: {
            "1_image": {
                "cols_names": "1_image",
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
                "type_processing": "categorical"
            }
        }
    },
    "fashion_mnist": {
        1: {
            "1_image": {
                "cols_names": "1_image",
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
                "type_processing": "categorical"
            }
        }
    },
    "cifar10": {
        1: {
            "1_image": {
                "cols_names": "1_image",
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
                "type_processing": "categorical"
            }
        }
    },
    "cifar100": {
        1: {
            "1_image": {
                "cols_names": "1_image",
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
                "type_processing": "categorical"
            }
        }
    }
}
# Конфиги
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
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "use_generator": False,
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "group": DatasetGroupChoice.keras,
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
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "use_generator": False,
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "group": DatasetGroupChoice.keras,
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
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "use_generator": False,
            },
            {
                "alias": "cifar100",
                "name": "Сifar 100",
                "group": DatasetGroupChoice.keras,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (32, 32, 3),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Image,
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
                        "task": LayerOutputTypeChoice.Classification,
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
                            "task": LayerInputTypeChoice.Image,
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
                            "task": LayerOutputTypeChoice.Classification,
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
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
                "use_generator": False,
            }
        ],
    },
    {
        "alias": "terra",
        "name": "Terra",
        "datasets": [
            {
                "alias": "sberbank_timeseries",
                "name": "Акции сбербанка",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "shape": (4, 30),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Dataframe,
                        "classes_names": [],
                        "num_classes": 1,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "shape": (2, 2),
                        "name": "Выход 1",
                        "task": LayerOutputTypeChoice.Timeseries,
                        "classes_names": [],
                        "num_classes": 1,
                    },
                },
                "columns": {
                    1: {
                        "1_1": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_2": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_3": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_4": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                    },
                    2: {
                        "2_3": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Output 1",
                            "num_classes": 1,
                            "shape": (2,),
                            "task": "Scaler",
                        },
                        "2_4": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Output 1",
                            "num_classes": 1,
                            "shape": (2,),
                            "task": "Scaler",
                        },
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "cars_30_classes",
                "name": "Автомобили (30 классов)",
                "group": DatasetGroupChoice.terra,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (128, 160, 3),
                        "name": "Изображения автомобилей",
                        "task": LayerInputTypeChoice.Image,
                        "num_classes": 30,
                        "classes_names": [
                            "Audi",
                            "BMW",
                            "Bentley",
                            "Cadillac",
                            "Chevrolet",
                            "Citroen",
                            "Dodge",
                            "Fiat",
                            "Ford",
                            "GMC",
                            "Honda",
                            "Infiniti",
                            "Jaguar",
                            "Jeep",
                            "KIA",
                            "Land_Rover",
                            "Lexus",
                            "Mazda",
                            "Mercedes_Benz",
                            "Nissan",
                            "Opel",
                            "Peugeot",
                            "Porsche",
                            "Renault",
                            "Rolls_Royce",
                            "Skoda",
                            "Subaru",
                            "Toyota",
                            "Volkswagen",
                            "Volvo",
                        ],
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (30,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "num_classes": 30,
                        "classes_names": [
                            "Audi",
                            "BMW",
                            "Bentley",
                            "Cadillac",
                            "Chevrolet",
                            "Citroen",
                            "Dodge",
                            "Fiat",
                            "Ford",
                            "GMC",
                            "Honda",
                            "Infiniti",
                            "Jaguar",
                            "Jeep",
                            "KIA",
                            "Land_Rover",
                            "Lexus",
                            "Mazda",
                            "Mercedes_Benz",
                            "Nissan",
                            "Opel",
                            "Peugeot",
                            "Porsche",
                            "Renault",
                            "Rolls_Royce",
                            "Skoda",
                            "Subaru",
                            "Toyota",
                            "Volkswagen",
                            "Volvo",
                        ],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_image": {
                            "datatype": "2D",
                            "dtype": "float32",
                            "name": "Изображения автомобилей",
                            "shape": (128, 160, 3),
                            "task": LayerInputTypeChoice.Image,
                            "num_classes": 30,
                            "classes_names": [
                                "Audi",
                                "BMW",
                                "Bentley",
                                "Cadillac",
                                "Chevrolet",
                                "Citroen",
                                "Dodge",
                                "Fiat",
                                "Ford",
                                "GMC",
                                "Honda",
                                "Infiniti",
                                "Jaguar",
                                "Jeep",
                                "KIA",
                                "Land_Rover",
                                "Lexus",
                                "Mazda",
                                "Mercedes_Benz",
                                "Nissan",
                                "Opel",
                                "Peugeot",
                                "Porsche",
                                "Renault",
                                "Rolls_Royce",
                                "Skoda",
                                "Subaru",
                                "Toyota",
                                "Volkswagen",
                                "Volvo",
                            ],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                    2: {
                        "2_classification": {
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "name": "Метки классов",
                            "shape": (30,),
                            "task": LayerOutputTypeChoice.Classification,
                            "num_classes": 30,
                            "classes_names": [
                                "Audi",
                                "BMW",
                                "Bentley",
                                "Cadillac",
                                "Chevrolet",
                                "Citroen",
                                "Dodge",
                                "Fiat",
                                "Ford",
                                "GMC",
                                "Honda",
                                "Infiniti",
                                "Jaguar",
                                "Jeep",
                                "KIA",
                                "Land_Rover",
                                "Lexus",
                                "Mazda",
                                "Mercedes_Benz",
                                "Nissan",
                                "Opel",
                                "Peugeot",
                                "Porsche",
                                "Renault",
                                "Rolls_Royce",
                                "Skoda",
                                "Subaru",
                                "Toyota",
                                "Volkswagen",
                                "Volvo",
                            ],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                },
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "use_generator": False,
            },
            {
                "alias": "cars",
                "name": "Автомобили",
                "group": DatasetGroupChoice.terra,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (120, 176, 3),
                        "name": "Изображения автомобилей",
                        "task": LayerInputTypeChoice.Image,
                        "num_classes": 3,
                        "classes_names": ["Мерседес", "Рено", "Феррари"],
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (3,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "num_classes": 3,
                        "classes_names": ["Мерседес", "Рено", "Феррари"],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_image": {
                            "datatype": "2D",
                            "dtype": "float32",
                            "name": "Изображения автомобилей",
                            "shape": (120, 176, 3),
                            "task": LayerInputTypeChoice.Image,
                            "num_classes": 3,
                            "classes_names": ["Мерседес", "Рено", "Феррари"],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                    2: {
                        "2_classification": {
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "name": "Метки классов",
                            "shape": (3,),
                            "task": LayerOutputTypeChoice.Classification,
                            "num_classes": 3,
                            "classes_names": ["Мерседес", "Рено", "Феррари"],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                },
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "use_generator": False,
            },
            {
                "alias": "airplane",
                "name": "Самолеты",
                "group": DatasetGroupChoice.terra,
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (128, 160, 3),
                        "name": "Изображения самолетов",
                        "task": LayerInputTypeChoice.Image,
                        "num_classes": 1,
                        "classes_names": ["Самолеты"],
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "shape": (128, 160, 2),
                        "name": "Маски сегментации",
                        "task": LayerOutputTypeChoice.Segmentation,
                        "num_classes": 2,
                        "classes_names": ["Небо", "Самолет"],
                        "classes_colors": ["black", "red"],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "columns": {
                    1: {
                        "1_image": {
                            "datatype": "2D",
                            "dtype": "float32",
                            "shape": (128, 160, 3),
                            "name": "Изображения самолетов",
                            "task": LayerInputTypeChoice.Image,
                            "num_classes": 1,
                            "classes_names": ["Самолеты"],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                    2: {
                        "2_classification": {
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "shape": (128, 160, 2),
                            "name": "Маски сегментации",
                            "task": LayerOutputTypeChoice.Segmentation,
                            "num_classes": 2,
                            "classes_names": ["Небо", "Самолет"],
                            "encoding": LayerEncodingChoice.ohe,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "lips",
                "name": "Губы",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (120, 176, 3),
                        "name": "Изображения",
                        "task": LayerInputTypeChoice.Image,
                        "classes_names": ["Оригинальные изображения"],
                        "num_classes": 1,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "2D",
                        "dtype": "uint8",
                        "shape": (120, 176, 2),
                        "name": "Маски сегментации",
                        "task": LayerOutputTypeChoice.Segmentation,
                        "classes_names": ["Фон", "Губы"],
                        "classes_colors": ["black", "lime"],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_image": {
                            "classes_names": ["Оригинальные изображения"],
                            "datatype": "2D",
                            "dtype": "float32",
                            "encoding": "none",
                            "name": "Изображения",
                            "num_classes": 1,
                            "shape": (120, 176, 3),
                            "task": LayerInputTypeChoice.Image,
                        }
                    },
                    2: {
                        "2_segmentation": {
                            "classes_colors": ["black", "lime"],
                            "classes_names": ["Лицо", "Губы"],
                            "datatype": "2D",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Маски сегментации",
                            "num_classes": 2,
                            "shape": (120, 176, 2),
                            "task": LayerOutputTypeChoice.Segmentation,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "bus_passengers",
                "name": "Пассажиры автобусов",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (200, 100, 3),
                        "name": "Изображения",
                        "task": LayerInputTypeChoice.Image,
                        "classes_names": ["Входящий", "Выходящий"],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (2,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "num_classes": 2,
                        "classes_names": ["Входящий", "Выходящий"],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_image": {
                            "classes_names": ["Входящий", "Выходящий"],
                            "datatype": "2D",
                            "dtype": "float32",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Изображения",
                            "num_classes": 2,
                            "shape": (200, 100, 3),
                            "task": LayerInputTypeChoice.Image,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": ["Входящий", "Выходящий"],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метки классов",
                            "num_classes": 2,
                            "shape": (2,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "heads",
                "name": "Пассажиры автобусов (попарно)",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (100, 100, 3),
                        "name": "Пассажир 1",
                        "task": LayerInputTypeChoice.Image,
                        "classes_names": ["Heads.csv"],
                        "num_classes": 1,
                        "encoding": LayerEncodingChoice.none,
                    },
                    2: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (100, 100, 3),
                        "name": "Пассажир 2",
                        "task": LayerInputTypeChoice.Image,
                        "classes_names": ["Heads.csv"],
                        "num_classes": 1,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    3: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (2,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "num_classes": 2,
                        "classes_names": ["Не совпадают", "Совпадают"],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_First image": {
                            "classes_names": ["Heads.csv"],
                            "datatype": "2D",
                            "dtype": "float32",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Пассажир 1",
                            "num_classes": 1,
                            "shape": (100, 100, 3),
                            "task": LayerInputTypeChoice.Image,
                        }
                    },
                    2: {
                        "2_Second image": {
                            "classes_names": ["Heads.csv"],
                            "datatype": "2D",
                            "dtype": "float32",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Пассажир 2",
                            "num_classes": 1,
                            "shape": (100, 100, 3),
                            "task": LayerInputTypeChoice.Image,
                        }
                    },
                    3: {
                        "3_Label": {
                            "classes_names": ["Не совпадают", "Совпадают"],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метка класса",
                            "num_classes": 2,
                            "shape": (2,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "marki_moloka",
                "name": "Марки молока",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "2D",
                        "dtype": "float32",
                        "shape": (200, 100, 3),
                        "name": "Изображения",
                        "task": LayerInputTypeChoice.Image,
                        "classes_names": [
                            "Parmalat",
                            "Кубанская бурёнка",
                            "Семейный формат",
                        ],
                        "num_classes": 3,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (3,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "num_classes": 2,
                        "classes_names": [
                            "Parmalat",
                            "Кубанская бурёнка",
                            "Семейный формат",
                        ],
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_image": {
                            "classes_names": [
                                "Parmalat",
                                "Кубанская бурёнка",
                                "Семейный формат",
                            ],
                            "datatype": "2D",
                            "dtype": "float32",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Изображения",
                            "num_classes": 2,
                            "shape": (200, 100, 3),
                            "task": LayerInputTypeChoice.Image,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": [
                                "Parmalat",
                                "Кубанская бурёнка",
                                "Семейный формат",
                            ],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метки классов",
                            "num_classes": 3,
                            "shape": (3,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "symptoms",
                "name": "Симптомы",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "uint32",
                        "shape": (100,),
                        "name": "Симптомы",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": [
                            "Аппендицит",
                            "Гастрит",
                            "Гепатит",
                            "Дуоденит",
                            "Колит",
                            "Панкреатит",
                            "Холецистит",
                            "Эзофагит",
                            "Энтерит",
                            "Язва",
                        ],
                        "num_classes": 10,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (10,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": [
                            "Аппендицит",
                            "Гастрит",
                            "Гепатит",
                            "Дуоденит",
                            "Колит",
                            "Панкреатит",
                            "Холецистит",
                            "Эзофагит",
                            "Энтерит",
                            "Язва",
                        ],
                        "num_classes": 10,
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_text": {
                            "classes_names": [
                                "Аппендицит",
                                "Гастрит",
                                "Гепатит",
                                "Дуоденит",
                                "Колит",
                                "Панкреатит",
                                "Холецистит",
                                "Эзофагит",
                                "Энтерит",
                                "Язва",
                            ],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Симптомы",
                            "num_classes": 10,
                            "shape": (100,),
                            "task": LayerInputTypeChoice.Text,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": [
                                "Аппендицит",
                                "Гастрит",
                                "Гепатит",
                                "Дуоденит",
                                "Колит",
                                "Панкреатит",
                                "Холецистит",
                                "Эзофагит",
                                "Энтерит",
                                "Язва",
                            ],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метки классов",
                            "num_classes": 10,
                            "shape": (10,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "writers",
                "name": "Тексты писателей",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "shape": (1000,),
                        "name": "Тексты писателей",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": [
                            "Булгаков",
                            "Клиффорд Саймак",
                            "Макс Фрай",
                            "О. Генри",
                            "Рэй Брэдберри",
                            "Стругацкие"
                        ],
                        "num_classes": 6,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (6,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": [
                            "Булгаков",
                            "Клиффорд Саймак",
                            "Макс Фрай",
                            "О. Генри",
                            "Рэй Брэдберри",
                            "Стругацкие"
                        ],
                        "num_classes": 6,
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_text": {
                            "classes_names": [
                                "Булгаков",
                                "Клиффорд Саймак",
                                "Макс Фрай",
                                "О. Генри",
                                "Рэй Брэдберри",
                                "Стругацкие"
                            ],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Тексты писателей",
                            "num_classes": 6,
                            "shape": (1000,),
                            "task": LayerInputTypeChoice.Text,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": [
                                "Булгаков",
                                "Клиффорд Саймак",
                                "Макс Фрай",
                                "О. Генри",
                                "Рэй Брэдберри",
                                "Стругацкие"
                            ],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метки классов",
                            "num_classes": 6,
                            "shape": (6,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "tesla",
                "name": "Отзывы на Теслу",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "shape": (100,),
                        "name": "Отзывы",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": [
                            "Негативные",
                            "Позитивные"
                        ],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (2,),
                        "name": "Метки классов",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": [
                            "Негативные",
                            "Позитивные"
                        ],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_text": {
                            "classes_names": [
                                "Негативные",
                                "Позитивные"
                            ],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Отзывы",
                            "num_classes": 2,
                            "shape": (100,),
                            "task": LayerInputTypeChoice.Text,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": [
                                "Негативные",
                                "Позитивные"
                            ],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Метки классов",
                            "num_classes": 2,
                            "shape": (2,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "rezjume",
                "name": "Резюме",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "shape": (20,),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": [
                            "да",
                            "нет"
                        ],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (2,),
                        "name": "Выход 2",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": [
                            "да",
                            "нет"
                        ],
                        "num_classes": 2,
                        "encoding": LayerEncodingChoice.ohe,
                    },
                },
                "columns": {
                    1: {
                        "1_text": {
                            "classes_names": [
                                "да",
                                "нет"
                            ],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Отзывы",
                            "num_classes": 2,
                            "shape": (20,),
                            "task": LayerInputTypeChoice.Text,
                        }
                    },
                    2: {
                        "2_classification": {
                            "classes_names": [
                                "да",
                                "нет"
                            ],
                            "datatype": "DIM",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.ohe,
                            "name": "Выход 2",
                            "num_classes": 2,
                            "shape": (2,),
                            "task": LayerOutputTypeChoice.Classification,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "docs",
                "name": "Договоры",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.text_segmentation,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "shape": (100,),
                        "name": "Договора",
                        "task": LayerInputTypeChoice.Text,
                        "encoding": LayerEncodingChoice.none,
                        "classes_names": ["Договора432"],
                        "num_classes": 1,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "1D",
                        "dtype": "uint8",
                        "shape": (100, 6),
                        "name": "Сегментация договоров",
                        "task": LayerOutputTypeChoice.TextSegmentation,
                        "encoding": LayerEncodingChoice.multi,
                        "classes_names": [
                            "<s1>",
                            "<s2>",
                            "<s3>",
                            "<s4>",
                            "<s5>",
                            "<s6>",
                        ],
                        "num_classes": 6,
                    },
                },
                "columns": {
                    1: {
                        "1_text": {
                            "classes_names": ["Договора432"],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Договора",
                            "num_classes": 1,
                            "shape": [100],
                            "task": LayerInputTypeChoice.Text,
                        }
                    },
                    "2": {
                        "2_text_segmentation": {
                            "classes_names": [
                                "<s1>",
                                "<s2>",
                                "<s3>",
                                "<s4>",
                                "<s5>",
                                "<s6>",
                            ],
                            "datatype": "1D",
                            "dtype": "uint8",
                            "encoding": LayerEncodingChoice.multi,
                            "name": "Сегментация договоров",
                            "num_classes": 6,
                            "shape": [100, 6],
                            "task": LayerOutputTypeChoice.TextSegmentation,
                        }
                    },
                },
                "use_generator": False,
            },
            # {
            #     "alias": "smart_home",
            #     "name": "Умный дом",
            #     "group": DatasetGroupChoice.terra,
            #     "inputs": {
            #         1: {
            #             # "datatype": "2D",
            #             # "dtype": "float32",
            #             # "shape": (),  # TODO
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Audio,
            #             "num_classes": 4,
            #             "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
            #             "encoding": LayerEncodingChoice.none
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (4,),
            #             "name": "Метки классов",
            #             "task": LayerOutputTypeChoice.Classification,
            #             "num_classes": 4,
            #             "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
            #             "encoding": LayerEncodingChoice.ohe
            #         },
            #     },
            #     "columns": {1: {"1_image": {"datatype": "2D",
            #                                 "dtype": "float32",
            #                                 "name": "Изображения автомобилей",
            #                                 "shape": (120, 176, 3),
            #                                 "task": LayerInputTypeChoice.Image,
            #                                 "num_classes": 3,
            #                                 "classes_names": ["Мерседес", "Рено", "Феррари"],
            #                                 "encoding": LayerEncodingChoice.ohe
            #                                 }
            #                     },
            #                 2: {"2_classification": {"datatype": "DIM",
            #                                          "dtype": "uint8",
            #                                          "name": "Метки классов",
            #                                          "shape": (4,),
            #                                          "task": LayerOutputTypeChoice.Classification,
            #                                          "num_classes": 4,
            #                                          "classes_names": ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"],
            #                                          "encoding": LayerEncodingChoice.ohe
            #                                          }
            #                     }
            #                 },
            #     "tags": [
            #         Tags.audio,
            #         Tags.classification,
            #         Tags.terra_ai,
            #     ],
            #     "use_generator": False,
            # },
            {
                "alias": "trading",
                "name": "Трейдинг",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.trading,
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "1D",
                        "dtype": "float32",
                        "shape": (4, 30),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Dataframe,
                        "classes_names": [],
                        "num_classes": 1,
                        "encoding": LayerEncodingChoice.none,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "int64",
                        "shape": (1,),
                        "name": "Выход 1",
                        "task": LayerOutputTypeChoice.Timeseries,
                        "num_classes": 3,
                        "classes_names": ["Не изменился", "Вверх", "Вниз"],
                    },
                },
                "columns": {
                    1: {
                        "1_<CLOSE>": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_<HIGH>": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_<LOW>": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                        "1_<OPEN>": {
                            "datatype": "DIM",
                            "dtype": "float64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Input 1",
                            "num_classes": 1,
                            "shape": (30,),
                            "task": "Scaler",
                        },
                    },
                    2: {
                        "2_<CLOSE>": {
                            "classes_names": ["Не изменился", "Вверх", "Вниз"],
                            "datatype": "DIM",
                            "dtype": "int64",
                            "encoding": LayerEncodingChoice.none,
                            "name": "Output 1",
                            "num_classes": 3,
                            "shape": (1,),
                            "task": LayerOutputTypeChoice.Timeseries,
                        }
                    },
                },
                "use_generator": False,
            },
            {
                "alias": "flats",
                "name": "Квартиры",
                "group": DatasetGroupChoice.terra,
                "tags": [
                    Tags.text,
                    Tags.regression,
                    Tags.russian,
                    Tags.terra_ai,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "uint8",
                        "shape": (1010,),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Dataframe,
                        "encoding": LayerEncodingChoice.none,
                        "num_classes": 1010,
                        "classes_names": [],
                    },
                    2: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "shape": (5000,),
                        "name": "Описание квартиры",
                        "task": LayerInputTypeChoice.Text,
                        "encoding": LayerEncodingChoice.none,
                        "num_classes": 1,
                        "classes_names": ["flats.csv"],
                    },
                },
                "outputs": {
                    3: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "shape": (1,),
                        "name": "Цена квартиры",
                        "task": LayerOutputTypeChoice.Regression,
                        "encoding": LayerEncodingChoice.none,
                        "num_classes": 1,
                        "classes_names": ["flats.csv"],
                    },
                },
                "use_generator": False,
            },
        ],
    },
    {
        "alias": "custom",
        "name": "Собственные",
    },
]
