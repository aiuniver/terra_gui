import cv2
import numpy as np
import pandas as pd
import tensorflow

from tensorflow.keras.utils import load_img
from typing import Any

from terra_ai.cascades.main_blocks import CascadeBlock, BaseBlock


class BaseOutput(BaseBlock):

    cascade_input = None
    output_file = None

    def set_inputs(self, cascade_input):
        self.cascade_input = cascade_input

    def set_out(self, output_file):
        self.output_file = output_file

    def execute(self):
        pass


class ImageOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        pass


class TextOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        source = self.cascade_input.prepare_sources()
        self.cascade_input.set_source(source)
        with open(self.output_file, 'a') as f:
            f.write(str(list(self.inputs.values())[0].execute()) + '\n')
        return self.output_file


class AudioOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        pass


class DataframeOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        pass


class VideoOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self):
        return list(self.inputs.values())[0].execute()


class VideoFrameOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()
        self.shape = (kwargs.get('width'), kwargs.get('height'))

    def execute(self):
        out = []

        writer = cv2.VideoWriter(
            self.output_file, cv2.VideoWriter_fourcc(*"VP80"), 30, self.shape
        )
        sources = self.cascade_input.prepare_sources(self.shape)

        for source in sources:
            self.cascade_input.set_source(source)
            frame = list(self.inputs.values())[0].execute()
            if len(frame.shape) == 4:
                for i in frame:
                    writer.write(frame(i))
            else:
                img = tensorflow.image.resize(frame, self.shape[::-1]).numpy()
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img)
        writer.release()
        return self.output_file


class Output(CascadeBlock):

    Image = ImageOutput
    Text = TextOutput
    Audio = AudioOutput
    Dataframe = DataframeOutput
    Video = VideoFrameOutput
    video_by_frame = VideoFrameOutput
