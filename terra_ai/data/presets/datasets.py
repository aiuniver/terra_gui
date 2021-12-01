"""
Предустановки датасетов
"""

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


VersionsGroup = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "mnist": [{"alias": "default",
                           "name": "Стандартная",
                           "date": datetime.now().isoformat(),
                           "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                           "use_generator": False,
                           "inputs": {
                               "1": {"name": "Изображения цифр",
                                     "datatype": "1D",
                                     "dtype": "uint8",
                                     "shape": [28, 28],
                                     "num_classes": 10,
                                     "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                     "classes_colors": None,
                                     "encoding": "none",
                                     "task": "Image"}
                           },
                           "outputs": {
                                "2": {"name": "Метки классов",
                                      "datatype": "DIM",
                                      "dtype": "uint8",
                                      "shape": [10],
                                      "num_classes": 10,
                                      "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                      "classes_colors": None,
                                      "encoding": "ohe",
                                      "task": "Classification"}
                           },
                           "service": {},
                           "columns": {
                               "1": {
                                   "1_image": {"name": "Изображения цифр",
                                               "datatype": "1D",
                                               "dtype": "float32",
                                               "shape": [28, 28],
                                               "num_classes": 10,
                                               "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                               "classes_colors": None,
                                               "encoding": "none",
                                               "task": "Image"
                                               }
                               },
                               "2": {
                                   "2_classification": {"name": "Метки классов",
                                                        "datatype": "DIM",
                                                        "dtype": "uint8",
                                                        "shape": [10],
                                                        "num_classes": 10,
                                                        "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                                        "classes_colors": None,
                                                        "encoding": "ohe",
                                                        "task": "Classification"
                                                        }
                               }
                           }
                           },
                          {"alias": "add_dimension",
                           "name": "Добавленная размерность",
                           "date": datetime.now().isoformat(),
                           "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                           "use_generator": False,
                           "inputs": {
                               "1": {"name": "Изображения цифр",
                                     "datatype": "2D",
                                     "dtype": "float32",
                                     "shape": [28, 28, 1],
                                     "num_classes": 10,
                                     "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                     "classes_colors": None,
                                     "encoding": "none",
                                     "task": "Image"}
                           },
                           "outputs": {
                               "2": {"name": "Метки классов",
                                     "datatype": "DIM",
                                     "dtype": "uint8",
                                     "shape": [10],
                                     "num_classes": 10,
                                     "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                     "classes_colors": None,
                                     "encoding": "ohe",
                                     "task": "Classification"}
                           },
                           "service": {},
                           "columns": {
                               "1": {
                                   "1_image": {"name": "Изображения цифр",
                                               "datatype": "2D",
                                               "dtype": "float32",
                                               "shape": [28, 28, 1],
                                               "num_classes": 10,
                                               "classes_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                               "classes_colors": None,
                                               "encoding": "none",
                                               "task": "Image"
                                               }
                               },
                               "2": {
                                   "2_classification": {"name": "Метки классов",
                                                        "datatype": "DIM",
                                                        "dtype": "uint8",
                                                        "shape": [10],
                                                        "num_classes": 10,
                                                        "classes_names": ["0", "1", "2", "3", "4",
                                                                          "5", "6", "7", "8", "9"],
                                                        "classes_colors": None,
                                                        "encoding": "ohe",
                                                        "task": "Classification"
                                                        }
                               }
                           }
                           }],
                "fashion_mnist": [{"alias": "default",
                                   "name": "Стандартная",
                                   "date": datetime.now().isoformat(),
                                   "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                                   "use_generator": False,
                                   "inputs": {
                                       "1": {"name": "Изображения одежды",
                                             "datatype": "1D",
                                             "dtype": "uint8",
                                             "shape": [28, 28],
                                             "num_classes": 10,
                                             "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                                               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
                                             "classes_colors": None,
                                             "encoding": "none",
                                             "task": "Image"}
                                   },
                                   "outputs": {
                                       "2": {"name": "Метки классов",
                                             "datatype": "DIM",
                                             "dtype": "uint8",
                                             "shape": [10],
                                             "num_classes": 10,
                                             "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                                               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
                                             "classes_colors": None,
                                             "encoding": "ohe",
                                             "task": "Classification"}
                                   },
                                   "service": {},
                                   "columns": {
                                       "1": {
                                           "1_image": {"name": "Изображения одежды",
                                                       "datatype": "1D",
                                                       "dtype": "float32",
                                                       "shape": [28, 28],
                                                       "num_classes": 10,
                                                       "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress",
                                                                         "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                                                                         "Ankle boot"],
                                                       "classes_colors": None,
                                                       "encoding": "none",
                                                       "task": "Image"
                                                       }
                                       },
                                       "2": {
                                           "2_classification": {"name": "Метки классов",
                                                                "datatype": "DIM",
                                                                "dtype": "uint8",
                                                                "shape": [10],
                                                                "num_classes": 10,
                                                                "classes_names": ["T-shirt/top", "Trouser", "Pullover",
                                                                                  "Dress", "Coat", "Sandal", "Shirt",
                                                                                  "Sneaker", "Bag", "Ankle boot"],
                                                                "classes_colors": None,
                                                                "encoding": "ohe",
                                                                "task": "Classification"
                                                                }
                                       }
                                   }
                                   },
                                  {"alias": "add_dimension",
                                   "name": "Добавленная размерность",
                                   "date": datetime.now().isoformat(),
                                   "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                                   "use_generator": False,
                                   "inputs": {
                                       "1": {"name": "Изображения цифр",
                                             "datatype": "2D",
                                             "dtype": "float32",
                                             "shape": [28, 28, 1],
                                             "num_classes": 10,
                                             "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                                               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
                                             "classes_colors": None,
                                             "encoding": "none",
                                             "task": "Image"}
                                   },
                                   "outputs": {
                                       "2": {"name": "Метки классов",
                                             "datatype": "DIM",
                                             "dtype": "uint8",
                                             "shape": [10],
                                             "num_classes": 10,
                                             "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                                                               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
                                             "classes_colors": None,
                                             "encoding": "ohe",
                                             "task": "Classification"}
                                   },
                                   "service": {},
                                   "columns": {
                                       "1": {
                                           "1_image": {"name": "Изображения цифр",
                                                       "datatype": "2D",
                                                       "dtype": "float32",
                                                       "shape": [28, 28, 1],
                                                       "num_classes": 10,
                                                       "classes_names": ["T-shirt/top", "Trouser", "Pullover", "Dress",
                                                                         "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                                                                         "Ankle boot"],
                                                       "classes_colors": None,
                                                       "encoding": "none",
                                                       "task": "Image"
                                                       }
                                       },
                                       "2": {
                                           "2_classification": {"name": "Метки классов",
                                                                "datatype": "DIM",
                                                                "dtype": "uint8",
                                                                "shape": [10],
                                                                "num_classes": 10,
                                                                "classes_names": ["T-shirt/top", "Trouser", "Pullover",
                                                                                  "Dress", "Coat", "Sandal", "Shirt",
                                                                                  "Sneaker", "Bag", "Ankle boot"],
                                                                "classes_colors": None,
                                                                "encoding": "ohe",
                                                                "task": "Classification"
                                                                }
                                       }
                                   }
                                   }],
                "cifar10": [{"alias": "default",
                                   "name": "Стандартная",
                                   "date": datetime.now().isoformat(),
                                   "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                                   "use_generator": False,
                                   "inputs": {
                                       "1": {"name": "Изображения",
                                             "datatype": "2D",
                                             "dtype": "float32",
                                             "shape": [32, 32, 3],
                                             "num_classes": 10,
                                             "classes_names": ["airplane", "automobile", "bird", "cat", "deer",
                                                               "dog", "frog", "horse", "ship", "truck"],
                                             "classes_colors": None,
                                             "encoding": "none",
                                             "task": "Image"}
                                   },
                                   "outputs": {
                                       "2": {"name": "Метки классов",
                                             "datatype": "DIM",
                                             "dtype": "uint8",
                                             "shape": [10],
                                             "num_classes": 10,
                                             "classes_names": ["airplane", "automobile", "bird", "cat", "deer",
                                                               "dog", "frog", "horse", "ship", "truck"],
                                             "classes_colors": None,
                                             "encoding": "ohe",
                                             "task": "Classification"}
                                   },
                                   "service": {},
                                   "columns": {
                                       "1": {
                                           "1_image": {"name": "Изображения одежды",
                                                       "datatype": "1D",
                                                       "dtype": "float32",
                                                       "shape": [32, 32, 3],
                                                       "num_classes": 10,
                                                       "classes_names": ["airplane", "automobile", "bird", "cat",
                                                                         "deer", "dog", "frog", "horse", "ship",
                                                                         "truck"],
                                                       "classes_colors": None,
                                                       "encoding": "none",
                                                       "task": "Image"
                                                       }
                                       },
                                       "2": {
                                           "2_classification": {"name": "Метки классов",
                                                                "datatype": "DIM",
                                                                "dtype": "uint8",
                                                                "shape": [10],
                                                                "num_classes": 10,
                                                                "classes_names": ["airplane", "automobile", "bird",
                                                                                  "cat", "deer", "dog", "frog", "horse",
                                                                                  "ship", "truck"],
                                                                "classes_colors": None,
                                                                "encoding": "ohe",
                                                                "task": "Classification"
                                                                }
                                       }
                                   }
                             }],
                "cifar100": [{"alias": "default",
                              "name": "Стандартная",
                              "date": datetime.now().isoformat(),
                              "size": {"value": 0, "short": 0, "unit": "\u041a\u0431"},
                              "use_generator": False,
                              "inputs": {
                                  "1": {"name": "Изображения",
                                        "datatype": "2D",
                                        "dtype": "float32",
                                        "shape": [32, 32, 3],
                                        "num_classes": 100,
                                        "classes_names": ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed",
                                                          "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge",
                                                          "bus", "butterfly", "camel", "can", "castle", "caterpillar",
                                                          "cattle", "chair", "chimpanzee", "clock", "cloud",
                                                          "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
                                                          "dolphin", "elephant", "flatfish", "forest", "fox", "girl",
                                                          "hamster", "house", "kangaroo", "keyboard", "lamp",
                                                          "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
                                                          "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
                                                          "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
                                                          "pickup_truck", "pine_tree", "plain", "plate", "poppy",
                                                          "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
                                                          "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
                                                          "skyscraper", "snail", "snake", "spider", "squirrel",
                                                          "streetcar", "sunflower", "sweet_pepper", "table", "tank",
                                                          "telephone", "television", "tiger", "tractor", "train",
                                                          "trout", "tulip", "turtle", "wardrobe", "whale",
                                                          "willow_tree", "wolf", "woman", "worm"],
                                        "classes_colors": None,
                                        "encoding": "none",
                                        "task": "Image"}
                              },
                              "outputs": {
                                  "2": {"name": "Метки классов",
                                        "datatype": "DIM",
                                        "dtype": "uint8",
                                        "shape": [100],
                                        "num_classes": 100,
                                        "classes_names": ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed",
                                                          "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge",
                                                          "bus", "butterfly", "camel", "can", "castle", "caterpillar",
                                                          "cattle", "chair", "chimpanzee", "clock", "cloud",
                                                          "cockroach", "couch", "cra", "crocodile", "cup", "dinosaur",
                                                          "dolphin", "elephant", "flatfish", "forest", "fox", "girl",
                                                          "hamster", "house", "kangaroo", "keyboard", "lamp",
                                                          "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
                                                          "maple_tree", "motorcycle", "mountain", "mouse", "mushroom",
                                                          "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
                                                          "pickup_truck", "pine_tree", "plain", "plate", "poppy",
                                                          "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
                                                          "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
                                                          "skyscraper", "snail", "snake", "spider", "squirrel",
                                                          "streetcar", "sunflower", "sweet_pepper", "table", "tank",
                                                          "telephone", "television", "tiger", "tractor", "train",
                                                          "trout", "tulip", "turtle", "wardrobe", "whale",
                                                          "willow_tree", "wolf", "woman", "worm"],
                                        "classes_colors": None,
                                        "encoding": "ohe",
                                        "task": "Classification"}
                              },
                              "service": {},
                              "columns": {
                                  "1": {
                                      "1_image": {"name": "Изображения одежды",
                                                  "datatype": "1D",
                                                  "dtype": "float32",
                                                  "shape": [32, 32, 3],
                                                  "num_classes": 100,
                                                  "classes_names": ["apple", "aquarium_fish", "baby", "bear", "beaver",
                                                                    "bed",
                                                                    "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
                                                                    "bridge",
                                                                    "bus", "butterfly", "camel", "can", "castle",
                                                                    "caterpillar",
                                                                    "cattle", "chair", "chimpanzee", "clock", "cloud",
                                                                    "cockroach", "couch", "cra", "crocodile", "cup",
                                                                    "dinosaur",
                                                                    "dolphin", "elephant", "flatfish", "forest", "fox",
                                                                    "girl",
                                                                    "hamster", "house", "kangaroo", "keyboard", "lamp",
                                                                    "lawn_mower", "leopard", "lion", "lizard",
                                                                    "lobster", "man",
                                                                    "maple_tree", "motorcycle", "mountain", "mouse",
                                                                    "mushroom",
                                                                    "oak_tree", "orange", "orchid", "otter",
                                                                    "palm_tree", "pear",
                                                                    "pickup_truck", "pine_tree", "plain", "plate",
                                                                    "poppy",
                                                                    "porcupine", "possum", "rabbit", "raccoon", "ray",
                                                                    "road",
                                                                    "rocket", "rose", "sea", "seal", "shark", "shrew",
                                                                    "skunk",
                                                                    "skyscraper", "snail", "snake", "spider",
                                                                    "squirrel",
                                                                    "streetcar", "sunflower", "sweet_pepper", "table",
                                                                    "tank",
                                                                    "telephone", "television", "tiger", "tractor",
                                                                    "train",
                                                                    "trout", "tulip", "turtle", "wardrobe", "whale",
                                                                    "willow_tree", "wolf", "woman", "worm"],
                                                  "classes_colors": None,
                                                  "encoding": "none",
                                                  "task": "Image"
                                                  }
                                  },
                                  "2": {
                                      "2_classification": {"name": "Метки классов",
                                                           "datatype": "DIM",
                                                           "dtype": "uint8",
                                                           "shape": [100],
                                                           "num_classes": 100,
                                                           "classes_names": ["apple", "aquarium_fish", "baby", "bear",
                                                                             "beaver", "bed",
                                                                             "bee", "beetle", "bicycle", "bottle",
                                                                             "bowl", "boy", "bridge",
                                                                             "bus", "butterfly", "camel", "can",
                                                                             "castle", "caterpillar",
                                                                             "cattle", "chair", "chimpanzee", "clock",
                                                                             "cloud",
                                                                             "cockroach", "couch", "cra", "crocodile",
                                                                             "cup", "dinosaur",
                                                                             "dolphin", "elephant", "flatfish",
                                                                             "forest", "fox", "girl",
                                                                             "hamster", "house", "kangaroo", "keyboard",
                                                                             "lamp",
                                                                             "lawn_mower", "leopard", "lion", "lizard",
                                                                             "lobster", "man",
                                                                             "maple_tree", "motorcycle", "mountain",
                                                                             "mouse", "mushroom",
                                                                             "oak_tree", "orange", "orchid", "otter",
                                                                             "palm_tree", "pear",
                                                                             "pickup_truck", "pine_tree", "plain",
                                                                             "plate", "poppy",
                                                                             "porcupine", "possum", "rabbit", "raccoon",
                                                                             "ray", "road",
                                                                             "rocket", "rose", "sea", "seal", "shark",
                                                                             "shrew", "skunk",
                                                                             "skyscraper", "snail", "snake", "spider",
                                                                             "squirrel",
                                                                             "streetcar", "sunflower", "sweet_pepper",
                                                                             "table", "tank",
                                                                             "telephone", "television", "tiger",
                                                                             "tractor", "train",
                                                                             "trout", "tulip", "turtle", "wardrobe",
                                                                             "whale",
                                                                             "willow_tree", "wolf", "woman", "worm"],
                                                           "classes_colors": None,
                                                           "encoding": "ohe",
                                                           "task": "Classification"
                                                           }
                                  }
                              }
                              }]
            }
        ]
    }
]


DatasetsGroups = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "alias": "mnist",
                "name": "Mnist",
                "group": "keras",
                "date": datetime.now().isoformat(),
                "architecture": "ImageClassification",
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "group": DatasetGroupChoice.keras.value,
                "date": datetime.now().isoformat(),
                "architecture": "ImageClassification",
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "group": DatasetGroupChoice.keras.value,
                "date": datetime.now().isoformat(),
                "architecture": "ImageClassification",
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
            },
            {
                "alias": "cifar100",
                "name": "Сifar 100",
                "group": DatasetGroupChoice.keras.value,
                "date": datetime.now().isoformat(),
                "architecture": "ImageClassification",
                "tags": [
                    Tags.image.value,
                    Tags.classification.value,
                    Tags.tensorflow_keras.value,
                ],
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
