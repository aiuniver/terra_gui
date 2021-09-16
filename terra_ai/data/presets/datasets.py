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
from terra_ai.data.training.extra import TaskChoice


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
                    },
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
                "alias": "imdb",
                "name": "Imdb",
                "group": DatasetGroupChoice.keras,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int",
                        "shape": (218,),  # шейп невозможно выявить. взял первый вектор
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": ["imdb"],
                        "num_classes": 1,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "float32",
                        "shape": (2,),
                        "name": "Выход 1",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": ["Отрицательный отзыв", "Положительный отзыв"],
                        "num_classes": 2,
                        "encoding": "ohe",
                    },
                },
            },
            {
                "alias": "boston_housing",
                "name": "Boston housing",
                "group": DatasetGroupChoice.keras,
                "tags": [
                    Tags.text,
                    Tags.regression,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "shape": (13,),
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": ["boston_housing"],
                        "num_classes": 1,
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "float64",
                        "shape": (1,),
                        "name": "Выход 1",
                        "task": LayerOutputTypeChoice.Regression,
                        "classes_names": ["Цена"],
                        "num_classes": 1,
                        "encoding": "none",
                    },
                },
            },
            {
                "alias": "reuters",
                "name": "Reuters",
                "group": DatasetGroupChoice.keras,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
                "inputs": {
                    1: {
                        "datatype": "DIM",
                        "dtype": "int",
                        "shape": (87,),  # шейп невозможно выявить. взял первый вектор
                        "name": "Вход 1",
                        "task": LayerInputTypeChoice.Text,
                        "classes_names": ["boston_housing"],
                        "num_classes": 1,
                        "encoding": "none",
                    },
                },
                "outputs": {
                    2: {
                        "datatype": "DIM",
                        "dtype": "float32",
                        "shape": (46,),
                        "name": "Выход 1",
                        "task": LayerOutputTypeChoice.Classification,
                        "classes_names": [
                            "cocoa",
                            "grain",
                            "veg-oil",
                            "earn",
                            "acq",
                            "wheat",
                            "copper",
                            "housing",
                            "money-supply",
                            "coffee",
                            "sugar",
                            "trade",
                            "reserves",
                            "ship",
                            "cotton",
                            "carcass",
                            "crude",
                            "nat-gas",
                            "cpi",
                            "money-fx",
                            "interest",
                            "gnp",
                            "meal-feed",
                            "alum",
                            "oilseed",
                            "gold",
                            "tin",
                            "strategic-metal",
                            "livestock",
                            "retail",
                            "ipi",
                            "iron-steel",
                            "rubber",
                            "heat",
                            "jobs",
                            "lei",
                            "bop",
                            "zinc",
                            "orange",
                            "pet-chem",
                            "dlr",
                            "gas",
                            "silver",
                            "wpi",
                            "hog",
                            "lead",
                        ],
                        "num_classes": 46,
                        "encoding": "ohe",
                    },
                },
            },
        ],
    },
    {
        "alias": "terra",
        "name": "Terra",
        "datasets": [
            # {
            #     "alias": "sber",  # TODO
            #     "name": "Sber",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.timeseries,
            #         Tags.terra_ai,
            #     ],
            #     "encoding": {2: "none"},
            #     "task_type": {2: TaskChoice.Timeseries},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (32, 32, 3),
            #             "name": "Вход 1",
            #             "task": "",
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (100,),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Timeseries,
            #         },
            #     },
            # },
            # {
            #     "alias": "cars",
            #     "name": "Автомобили",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.image,
            #         Tags.classification,
            #         Tags.terra_ai,
            #     ],
            #     "num_classes": {2: 3},
            #     "classes_names": {2: ["Мерседес", "Рено", "Феррари"]},
            #     "encoding": {2: "ohe"},
            #     "task_type": {2: TaskChoice.Classification},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (176, 220, 3),
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Image,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (3,),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Classification,
            #         },
            #     },
            # },
            # {
            #     "alias": "planes",
            #     "name": "Самолеты",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.image,
            #         Tags.segmentation,
            #         Tags.terra_ai,
            #     ],
            #     "num_classes": {2: 2},
            #     "classes_colors": {2: [[0, 0, 0], [255, 0, 0]]},
            #     "classes_names": {2: ["Небо", "Самолет"]},
            #     "encoding": {2: "ohe"},
            #     "task_type": {2: TaskChoice.Segmentation},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (176, 220, 3),
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Image,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "2D",
            #             "dtype": "uint8",
            #             "shape": (176, 220, 2),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Segmentation,
            #         },
            #     },
            # },
            # {
            #     "alias": "lips",
            #     "name": "Губы",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.image,
            #         Tags.segmentation,
            #         Tags.terra_ai,
            #     ],
            #     "num_classes": {2: 2},
            #     "classes_colors": {2: [[0, 0, 0], [0, 255, 0]]},
            #     "classes_names": {2: ["Фон", "Губы"]},
            #     "encoding": {2: "ohe"},
            #     "task_type": {2: TaskChoice.Segmentation},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (176, 220, 3),
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Image,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "2D",
            #             "dtype": "uint8",
            #             "shape": (176, 220, 2),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Segmentation,
            #         },
            #     },
            # },
            # {
            #     "alias": "diseases",
            #     "name": "Заболевания",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.text,
            #         Tags.classification,
            #         Tags.russian,
            #         Tags.terra_ai,
            #     ],
            #     "num_classes": {2: 10},
            #     "classes_names": {
            #         2: [
            #             "Аппендицит",
            #             "Гастрит",
            #             "Гепатит",
            #             "Дуоденит",
            #             "Колит",
            #             "Панкреатит",
            #             "Холецистит",
            #             "Эзофагит",
            #             "Энтерит",
            #             "Язва",
            #         ]
            #     },
            #     "encoding": {2: "ohe"},
            #     "task_type": {2: TaskChoice.Classification},
            #     "inputs": {
            #         1: {
            #             "datatype": "DIM",
            #             "dtype": "uint32",
            #             "shape": (100,),
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Text,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (10,),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Classification,
            #         },
            #     },
            # },
            # {
            #     "alias": "docs",
            #     "name": "Договоры",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.text,
            #         Tags.text_segmentation,
            #         Tags.russian,
            #         Tags.terra_ai,
            #     ],
            #     "encoding": {2: "multi"},
            #     "task_type": {2: TaskChoice.Segmentation},
            #     "inputs": {
            #         1: {
            #             "datatype": "DIM",
            #             "dtype": "uint32",
            #             "shape": (200,),
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Text,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "1D",
            #             "dtype": "uint8",
            #             "shape": (200, 6),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.TextSegmentation,
            #         },
            #     },
            # },
            # {
            #     "alias": "smart_home",
            #     "name": "Умный дом",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.audio,
            #         Tags.classification,
            #         Tags.smart_home,
            #         Tags.russian,
            #         Tags.terra_ai,
            #     ],
            #     "num_classes": {2: 4},
            #     "classes_names": {
            #         2: ["1_Кондиционер", "2_Свет", "3_Телевизор", "4_Шум"]
            #     },
            #     "encoding": {2: "ohe"},
            #     "task_type": {2: TaskChoice.Classification},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (),  # TODO
            #             "name": "Вход 1",
            #             "task": LayerInputTypeChoice.Audio,
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (4,),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Classification,
            #         },
            #     },
            # },
            # {
            #     "alias": "trading",  # TODO
            #     "name": "Трейдинг",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.trading,
            #         Tags.timeseries,
            #         Tags.terra_ai,
            #     ],
            #     "encoding": {2: "none"},
            #     "task_type": {2: "Timeseries"},
            #     "inputs": {
            #         1: {
            #             "datatype": "2D",
            #             "dtype": "float32",
            #             "shape": (),
            #             "name": "Вход 1",
            #             "task": "",
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (100,),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Timeseries,
            #         },
            #     },
            # },
            # {
            #     "alias": "flats",  # TODO
            #     "name": "Квартиры",
            #     "group": DatasetGroupChoice.terra,
            #     "tags": [
            #         Tags.text,
            #         Tags.regression,
            #         Tags.russian,
            #         Tags.terra_ai,
            #     ],
            #     "encoding": {2: "none"},
            #     "task_type": {2: "Regression"},
            #     "inputs": {
            #         1: {
            #             "datatype": "1D",
            #             "dtype": "float32",
            #             "shape": (),
            #             "name": "Вход 1",
            #             "task": "",
            #         },
            #     },
            #     "outputs": {
            #         2: {
            #             "datatype": "DIM",
            #             "dtype": "uint8",
            #             "shape": (),
            #             "name": "Выход 1",
            #             "task": LayerOutputTypeChoice.Regression,
            #         },
            #     },
            # },
        ],
    },
    {
        "alias": "custom",
        "name": "Собственные",
    },
]
