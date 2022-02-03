import cv2
import numpy as np
import pandas as pd
import tensorflow

from tensorflow.keras.utils import load_img, save_img
from typing import Any
from pydub import AudioSegment

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


class EmptyOutput(BaseOutput):

    def execute(self):
        source = self.cascade_input.prepare_sources()
        self.cascade_input.set_source(source)
        return list(self.inputs.values())[0].execute()


class ImageOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()
        self.result_type = ".webp"

    def execute(self):
        source = self.cascade_input.prepare_sources()
        self.cascade_input.set_source(source)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(path, img)

        # save_img(f"{self.output_file}{self.result_type}", image_)
        # return f"{self.output_file}{self.result_type}"
        # classes_names = params['classes_names'] if 'classes_names' in params.keys() else None
        #
        # def fun(acc):
        # acc = image_ * 100
        # # acc = np.array(image_)
        # # print(acc.shape)
        # if len(acc) == 1:
        #     acc = acc[0]
        # else:
        #     acc = np.mean(acc, axis=0)
        # print(acc)
        # acc = acc.round().astype(np.int)
        # print(acc, classes)
        # out = list(zip(classes, acc))
        # print(sorted(out, key=lambda x: x[-1], reverse=True))


class TextOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()
        self.result_type = ".txt"

    def execute(self):
        source = self.cascade_input.prepare_sources()
        self.cascade_input.set_source(source)
        with open(f"{self.output_file}{self.result_type}", 'a') as f:
            f.write(str(list(self.inputs.values())[0].execute()) + '\n')
        return f"{self.output_file}{self.result_type}"


class AudioOutput(BaseOutput):

    def __init__(self, **kwargs):
        super().__init__()
        self.result_type = ".webm"

    def execute(self):
        source = self.cascade_input.prepare_sources()
        self.cascade_input.set_source(source)
        audio = list(self.inputs.values())[0].execute()
        result = AudioSegment.from_file(audio, format="mp3")
        result.export(f"{self.output_file}{self.result_type}", format="webm")
        return f"{self.output_file}{self.result_type}"


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
        self.result_type = ".webm"

    def execute(self):
        out = []

        writer = cv2.VideoWriter(
            f"{self.output_file}{self.result_type}", cv2.VideoWriter_fourcc(*"VP80"), 30, self.shape
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

    Empty = EmptyOutput
    Image = ImageOutput
    Text = TextOutput
    Audio = AudioOutput
    Dataframe = DataframeOutput
    Video = VideoFrameOutput
    video_by_frame = VideoFrameOutput
