"""
Предустановки датасетов
"""

from enum import Enum


class Tags(dict, Enum):
    images = {"alias": "images", "name": "Images"}
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
        "alias": "preset",
        "name": "Предустановленные",
        "datasets": [
            {
                "alias": "mnist",
                "name": "Mnist",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "fashion_mnist",
                "name": "Fashion mnist",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar10",
                "name": "Cifar 10",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "cifar100",
                "name": "Cifar 100",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.tensorflow_keras,
                ],
            },
            {
                "alias": "imdb",
                "name": "IMDB",
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
                "tags": [
                    Tags.timeseries,
                    Tags.regression,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars",
                "name": "Автомобили",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "cars3",
                "name": "Автомобили 3",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "aircraft",
                "name": "Самолеты",
                "tags": [
                    Tags.images,
                    Tags.segmentation,
                    Tags.object_detection,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "lips",
                "name": "Губы",
                "tags": [
                    Tags.images,
                    Tags.segmentation,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "diseases",
                "name": "Заболевания",
                "tags": [
                    Tags.images,
                    Tags.classification,
                    Tags.russian,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "contracts",
                "name": "Договоры",
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
                "tags": [
                    Tags.trading,
                    Tags.timeseries,
                    Tags.terra_ai,
                ],
            },
            {
                "alias": "apartments",
                "name": "Квартиры",
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
