import cv2
import os
import json
from abc import ABC, abstractmethod

import numpy as np

from .main_blocks import CascadeBlock
from ..datasets.arrays_create import CreateArray


class BaseInput(ABC):

    def __init__(self):
        self.inputs: dict = {}
        self.sources: dict = {}
        self.dataset_path: str = ''

    def set_source(self, source):
        with open(os.path.join(self.dataset_path, 'config.json'), 'r') as cfg:
            data = json.load(cfg)
        inputs = data.get('inputs').keys()
        if len(inputs) == 1:
            column = data.get('columns', {}).keys()[0]
            self.sources = {inputs[0]: {column: source}}
        else:
            self.sources = None

    def set_dataset_path(self, path: str):
        self.dataset_path = path

    @abstractmethod
    def execute(self):
        pass


class ImageInput(BaseInput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class TextInput(BaseInput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class AudioInput(BaseInput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class DataframeInput(BaseInput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class VideoInput(BaseInput):

    def __init__(self):
        super().__init__()

    def execute(self):
        return self.sources[0]


class VideoFrameInput(BaseInput):

    def __init__(self):
        super().__init__()
        self.array_class = 'video'

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self):
        # array = CreateArray().execute(array_class=self.array_class,
        #                               dataset_path=self.dataset_path,
        #                               sources=self.sources)
        # return array
        cap = cv2.VideoCapture(self.sources[0])
        out_arr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out_arr.append(frame)
        return np.array(out_arr)

    def execute(self):
        return self.sources


class Input(CascadeBlock):

    Image = ImageInput
    Text = TextInput
    Audio = AudioInput
    Dataframe = DataframeInput
    Video = VideoInput
    VideoFrameInput = VideoFrameInput
