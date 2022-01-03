import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.utils import load_img
from typing import Any
from abc import ABC, abstractmethod


class BaseInput(ABC):

    @abstractmethod
    def execute(self, sources: Any):
        pass


class VideoInput(BaseInput):

    def execute(self, sources):
        if isinstance(sources, str):
            sources = [sources]

        for source in sources:
            yield source


class Input:

    video = VideoInput

    def get(self, input_type):
        return self.__getattribute__(input_type)()





class VideoFrameInput(Input):

    def execute(self, sources):
        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            yield frame