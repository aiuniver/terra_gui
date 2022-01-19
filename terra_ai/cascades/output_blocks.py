import cv2
import numpy as np
import pandas as pd
import tensorflow

from tensorflow.keras.utils import load_img
from typing import Any
from abc import ABC, abstractmethod

from terra_ai.cascades.main_blocks import CascadeBlock


class BaseOutput(ABC):

    def __init__(self):
        self.inputs: dict = {}

    @abstractmethod
    def execute(self):
        pass


class ImageOutput(BaseOutput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class TextOutput(BaseOutput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class AudioOutput(BaseOutput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class DataframeOutput(BaseOutput):

    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class VideoOutput(BaseOutput):

    def __init__(self):
        super().__init__()

    def execute(self):
        return list(self.inputs.values())[0].execute()


class VideoFrameOutput(BaseOutput):

    def __init__(self):
        super().__init__()
        self.cascade_input = None

    def set_inputs(self, cascade_input):
        self.cascade_input = cascade_input

    def execute(self):
        out = []
        shape = (1280, 720)

        writer = cv2.VideoWriter(
            "F:\\test.webm", cv2.VideoWriter_fourcc(*"VP80"), 30, shape
        )
        sources = self.cascade_input.prepare_sources()
        for source in sources:
            self.cascade_input.set_source(source)
            frame = list(self.inputs.values())[0].execute()
            if len(frame.shape) == 4:
                for i in frame:
                    writer.write(frame(i))
            else:
                img = tensorflow.image.resize(frame, shape[::-1]).numpy()
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img)
        writer.release()
        return writer


class Output(CascadeBlock):

    Image = ImageOutput
    Text = TextOutput
    Audio = AudioOutput
    Dataframe = DataframeOutput
    Video = VideoFrameOutput
    video_by_frame = VideoFrameOutput
