from enum import Enum
from pydantic import BaseModel


class SizeData(BaseModel):
    value: float
    unit: str


class DatasetSourceModeChoice(str, Enum):
    google_drive = "google_drive"
    url = "url"


class InputTypeChoice(str, Enum):
    images = "images"
    text = "text"
    audio = "audio"
    dataframe = "dataframe"


class LayerNetChoice(str, Enum):
    Convolutional = "Convolutional"
    Linear = "Linear"


class LayerScalerChoice(str, Enum):
    NoScaler = "NoScaler"
    MinMaxScaler = "MinMaxScaler"


class LayerPrepareMethodChoice(str, Enum):
    embedding = "embedding"
    bag_of_words = "bag_of_words"
    word_to_vec = "word_to_vec"
