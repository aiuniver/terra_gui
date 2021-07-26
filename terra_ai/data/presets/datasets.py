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
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar100",
                "name": "Cifar 100",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "imdb",
                "name": "IMDB",
                "limit": 1,
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
                "limit": 1,
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
                "limit": 1,
                "tags": [
                    Tags.text,
                    Tags.classification,
                    Tags.english,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "sber",
                "name": "Sber",
                "limit": 1,
                "tags": [
                    Tags.timeseries,
                    Tags.regression,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars",
                "name": "Автомобили",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars3",
                "name": "Автомобили 3",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "aircraft",
                "name": "Самолеты",
                "limit": 1,
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
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "diseases",
                "name": "Заболевания",
                "limit": 1,
                "tags": [
                    Tags.image,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "contracts",
                "name": "Договоры",
                "limit": 1,
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
                "limit": 1,
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
                "limit": 1,
                "tags": [
                    Tags.trading,
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "apartments",
                "name": "Квартиры",
                "limit": 1,
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
