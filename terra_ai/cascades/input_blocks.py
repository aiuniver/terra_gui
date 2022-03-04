import cv2
import os
import json

import numpy as np

from .main_blocks import CascadeBlock, BaseBlock
from ..datasets.arrays_create import CreateArray


class BaseInput(BaseBlock):

    def __init__(self):
        super().__init__()
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

    def execute(self):
        pass


class ImageInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()
        self.array_class = 'image'

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self):
        # params = {}
        # from terra_ai.data.datasets.creations.layers.input.types.Video import ParametersData
        # for key in ParametersData.__fields__.keys():
        #     params[key] = self.__dict__.get(key, None)
        # print(params)
        # array = CreateArray().execute_array(array_class=self.array_class,
        #                                     sources=self.sources, **params)
        return self.sources

    def execute(self):
        return self.sources[0]


class TextInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()
        self.array_class = 'text'

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self):
        # params = {}
        # from terra_ai.data.datasets.creations.layers.input.types.Audio import ParametersData
        # for key in ParametersData.__fields__.keys():
        #     params[key] = self.__dict__.get(key, None)
        # array = CreateArray().execute_array(array_class=self.array_class,
        #                                     sources=self.sources, **params)
        return self.sources

    def execute(self):
        return self.sources[0]


class AudioInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()
        self.array_class = 'audio'

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self):
        # params = {}
        # from terra_ai.data.datasets.creations.layers.input.types.Audio import ParametersData
        # for key in ParametersData.__fields__.keys():
        #     params[key] = self.__dict__.get(key, None)
        # array = CreateArray().execute_array(array_class=self.array_class,
        #                                     sources=self.sources, **params)
        return self.sources

    def execute(self):
        return self.sources[0]


class DataframeInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        return self.sources


class VideoInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self):
        return self.sources

    def execute(self):
        return self.sources[0]


class VideoFrameInput(BaseInput):

    def __init__(self, **kwargs):
        super().__init__()
        self.array_class = 'video'
        self.frame_mode = 'fit'
        self.video_mode = 'completely'
        self.max_frames = 300

    def set_source(self, source):
        self.sources = source

    def prepare_sources(self, shape):
        params = {}
        from terra_ai.data.datasets.creations.layers.input.types.Video import ParametersData
        for key in ParametersData.__fields__.keys():
            params[key] = self.__dict__.get(key, None)
        params['width'] = shape[0]
        params['height'] = shape[1]

        array = CreateArray().execute_array(array_class=self.array_class,
                                            sources=self.sources, **params)
        return np.squeeze(array, axis=0)
        # cap = cv2.VideoCapture(self.sources[0])
        # out_arr = []
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #
        #     out_arr.append(frame)
        # return np.array(out_arr)

    def execute(self):
        return self.sources


class Input(CascadeBlock):
    Image = ImageInput
    Text = TextInput
    Audio = AudioInput
    Dataframe = DataframeInput
    Video = VideoInput
    VideoFrameInput = VideoFrameInput
