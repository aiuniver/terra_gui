"""
Datasets constants data
"""
from enum import Enum
from typing import List, Any, Optional, Union, Dict

from pydantic import validator, DirectoryPath, FilePath, PositiveInt

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.types import StrictIntValueGe0, StrictFloatValueGe0
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
    instructions: List[Union[str, PositiveInt, Dict[str, List[Union[StrictIntValueGe0, StrictFloatValueGe0]]]]]
    parameters: Any

    # @validator("parameters", always=True)
    # def _validate_parameters(cls, value: Any, values, field) -> Any:
    #     return field.type_(**value or {})


class DatasetInstructionsData(BaseMixinData):
    inputs: Dict[PositiveInt, InstructionsData]
    outputs: Dict[PositiveInt, InstructionsData]


class PathsData(BaseMixinData):
    datasets: DirectoryPath
    instructions: Optional[DirectoryPath]
    arrays: DirectoryPath

