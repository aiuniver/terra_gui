"""
Datasets constants data
"""
from enum import Enum
from typing import List, Any

from pydantic import validator

from terra_ai.data.mixins import BaseMixinData
from terra_ai.datasets.arrays_create import CreateArray

DatasetArchives = {
    'трейдинг': 'trading.zip',
    'автомобили': 'cars.zip',
    'умный_дом': 'smart_home.zip',
    'квартиры': 'flats.zip',
    # 'диалоги': ['dialog.txt'],
    'автомобили_3': 'cars_3.zip',
    'заболевания': 'diseases.zip',
    'договоры': 'docs.zip',
    'самолеты': 'planes.zip',
    # 'болезни': ['origin.zip', 'segmentation.zip'],
    'губы': 'lips.zip',
    # 'жанры_музыки': ['genres.zip'],
    'sber': 'sber.zip'
}

DataType = {0: 'DIM',
            1: 'DIM',
            2: '1D',
            3: '2D',
            4: '3D',
            5: '4D'
            }


class Preprocesses(str, Enum):
    scaler = "scaler"
    tokenizer = "tokenizer"
    word2vec = "word2vec"
    augmentation = "augmentation"
    # tsgenerator = "tsgenerator"


class InstructionsData(BaseMixinData):
    sources: List[Any]
    parameters: Any

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})
