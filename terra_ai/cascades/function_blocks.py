import numpy as np
import tensorflow as tf
import cv2

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
from random import randrange

from terra_ai.cascades.input_blocks import Input
from terra_ai.cascades.main_blocks import CascadeBlock
from terra_ai.data.cascades.blocks.extra import ObjectDetectionFilterClassesList


class BaseFunction(ABC):

    def __init__(self):
        self.inputs: dict = {}

    @abstractmethod
    def execute(self):
        pass


class ChangeType(BaseFunction):

    def __init__(self, format_type):
        super().__init__()
        self.format_type = format_type

    def execute(self):
        array_ = list(self.inputs.values())[0].execute()
        return array_.astype(self.format_type)


class ChangeSize(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        image_ = list(self.inputs.values())[0].execute()
        return tf.image.resize(image_, self.format_size).numpy()


class MinMaxScale(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class CropImage(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class MaskedImage(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class PlotMaskSegmentation(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class PutTag(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class PostprocessBoxes(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class PlotBBoxes(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        self.classes = ObjectDetectionFilterClassesList  # kwargs.get("classes", [])
        self.colors = kwargs.get("colors", [])
        if not self.colors:
            self.colors = [tuple((randrange(1, 256) for _ in range(3))) for _ in range(len(self.classes))]
        self.line_thickness = kwargs.get("line_thickness", None)

    def execute(self):
        img, bboxes = None, None
        for input_type in self.inputs.keys():
            if input_type in Input.__dict__.keys():
                img = self.inputs.get(input_type).execute()
                print("PLOT IMG: ", img.shape)
            else:
                bboxes = self.inputs.get(input_type).execute()
                print("PLOT BBOX: ", bboxes, len(bboxes))

        # if img.any() and bboxes.any():
        img = np.squeeze(img)
        if bboxes.shape != (0, 5):
            brawl_color = True

            if bboxes.shape[-1] == 5:
                brawl_color = False

            # Plots one bounding box on image img
            tl = self.line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            if tl < 0:
                tl = 1
            for x in bboxes:
                cl = int(x[-1])
                if brawl_color:
                    color = self.colors[cl]
                else:
                    color = self.colors[0]
                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                cv2.rectangle(img, c1, c2, color, int(tl), cv2.LINE_AA)

                t_size = (114, 18)
                c2 = int(c1[0] + t_size[0] + 5 if (c1[0] + t_size[0] + 5) > 0 else 0), \
                     int(c1[1] - t_size[1] - 5 if (c1[1] - t_size[1] - 5) > 0 else 0)
                c1 = int(c1[0] if (c1[0]) > 0 else 0), int(c1[1] - 3 if (c1[1] - 3) > 0 else 0)

                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(pil_img)

                if brawl_color:
                    d.text((c1[0] + 5, c1[1] - 17), self.classes[cl], fill=(225, 255, 255, 0))
                else:
                    d.text((c1[0] + 5, c1[1] - 17), str(cl), fill=(225, 255, 255, 0))

                img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return img


class FilterClasses(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        # self.filter_classes = kwargs.get("filter_classes", [])
        self.filter_classes = [ObjectDetectionFilterClassesList.index(x)
                               for x in kwargs.get("filter_classes", [])]

    def execute(self):
        bboxes = list(self.inputs.values())[0].execute()
        filtered_bboxes = bboxes[bboxes[:, -1] == self.filter_classes[0]]

        for i in self.filter_classes[1:]:
            filtered_bboxes = np.concatenate((filtered_bboxes, bboxes[bboxes[:, -1] == i]))

        return filtered_bboxes


class Function(CascadeBlock):
    ChangeType = ChangeType
    ChangeSize = ChangeSize
    MinMaxScale = MinMaxScale
    CropImage = CropImage
    MaskedImage = MaskedImage
    PlotMaskSegmentation = PlotMaskSegmentation
    PutTag = PutTag
    PostprocessBoxes = PostprocessBoxes
    PlotBBoxes = PlotBBoxes
    FilterClasses = FilterClasses
