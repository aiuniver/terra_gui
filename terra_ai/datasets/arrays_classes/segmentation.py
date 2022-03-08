import os
import shutil
import numpy as np

from PIL import UnidentifiedImageError
from pydantic.color import Color
from tensorflow.keras.preprocessing.image import load_img
from typing import Any
from .base import Array

from terra_ai.datasets.utils import resize_frame


class SegmentationArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        image_list = []
        paths_list = []
        for elem in sources:
            try:
                load_img(elem).verify()
                image_list.append(elem)
            except (UnidentifiedImageError, IOError):
                pass

        if dataset_folder is not None:
            for elem in image_list:
                os.makedirs(os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem))), exist_ok=True)
                shutil.copyfile(elem, os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)),
                                                   os.path.basename(elem)))

            paths_list = [os.path.join(dataset_folder, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                          for elem in image_list]

        instructions = {'instructions': paths_list if paths_list else image_list,
                        'parameters': {'mask_range': options['mask_range'],
                                       'num_classes': len(options['classes_names']),
                                       'image_mode': options['image_mode'],
                                       'height': options['height'],
                                       'width': options['width'],
                                       'classes_colors': [Color(color).as_rgb_tuple() for color in
                                                          options['classes_colors']],
                                       'classes_names': options['classes_names'],
                                       'cols_names': options['cols_names'],
                                       'put': options['put']
                                       }
                        }

        return instructions

    def create(self, source: Any, **options):

        array = np.array(load_img(source, target_size=(options['height'], options['width'])))
        array = resize_frame(image_array=array,
                             target_shape=(options['height'], options['width']),
                             frame_mode=options['image_mode'])

        array = self.image_to_ohe(array, **options)

        return array

    def preprocess(self, array: np.ndarray, **options):

        return array

    @staticmethod
    def image_to_ohe(img_array, **options):
        mask_ohe = []
        for color in options['classes_colors']:
            color_array = np.expand_dims(np.where((color[0] + options['mask_range'] >= img_array[:, :, 0]) &
                                                  (img_array[:, :, 0] >= color[0] - options['mask_range']) &
                                                  (color[1] + options['mask_range'] >= img_array[:, :, 1]) &
                                                  (img_array[:, :, 1] >= color[1] - options['mask_range']) &
                                                  (color[2] + options['mask_range'] >= img_array[:, :, 2]) &
                                                  (img_array[:, :, 2] >= color[2] - options['mask_range']), 1, 0),
                                         axis=2)
            mask_ohe.append(color_array)

        return np.concatenate(np.array(mask_ohe), axis=2).astype(np.uint8)
