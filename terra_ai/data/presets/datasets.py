"""
Предустановки датасетов
"""

from enum import Enum


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
    audio = {"alias": "audio", "name": "Audio"}
    smart_home = {"alias": "smart_home", "name": "Smart home"}
    trading = {"alias": "trading", "name": "Trading"}


DatasetsGroups = [
    {
        "alias": "keras",
        "name": "Keras",
        "datasets": [
            {
                "alias": "mnist",
                "name": "Mnist",
                "group": "keras",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "group": "keras",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "group": "keras",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar100",
                "name": "Cifar 100",
                "group": "keras",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "imdb",
                "name": "IMDB",
                "group": "keras",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "boston_housing",
                "name": "Boston housing",
                "group": "keras",
                "tags": [
                    Tags.text,
                    Tags.regression,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "reuters",
                "name": "Reuters",
                "group": "keras",
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
            },
        ],
    },
    {
        "alias": "terra",
        "name": "Terra",
        "datasets": [
            {
                "alias": "sber",
                "name": "Sber",
                "group": "terra",
                "tags": [
                    Tags.timeseries,
                    Tags.regression,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars",
                "name": "Автомобили",
                "group": "terra",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars_3",
                "name": "Автомобили 3",
                "group": "terra",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "planes",
                "name": "Самолеты",
                "group": "terra",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "lips",
                "name": "Губы",
                "group": "terra",
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "diseases",
                "name": "Заболевания",
                "group": "terra",
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "docs",
                "name": "Договоры",
                "group": "terra",
                "tags": [
                    Tags.text,
                    Tags.segmentation,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "smart_home",
                "name": "Умный дом",
                "group": "terra",
                "tags": [
                    Tags.audio,
                    Tags.classification,
                    Tags.smart_home,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "trading",
                "name": "Трейдинг",
                "group": "terra",
                "tags": [
                    Tags.trading,
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "flats",
                "name": "Квартиры",
                "group": "terra",
                "tags": [
                    Tags.text,
                    Tags.regression,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
        ],
    },
    {
        "alias": "custom",
        "name": "Собственные",
    },
]
