import os
import numpy as np
import shutil
import imgaug

from PIL import UnidentifiedImageError
from tensorflow.keras.preprocessing.image import load_img

from terra_ai.datasets.utils import resize_frame
from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerNetChoice
from .base import Array


class ImageArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        p_list = []
        for elem in sources:
            try:
                load_img(elem)
                p_list.append(elem)
            except (UnidentifiedImageError, IOError):
                pass

        if dataset_folder is not None:
            p_list = self.save(p_list, target=dataset_folder)

        instructions = {'instructions': p_list,
                        'parameters': {'height': options['height'],
                                       'width': options['width'],
                                       'net': options['net'],
                                       'image_mode': options.get('image_mode', 'stretch'),
                                       'scaler': options['scaler'],
                                       'max_scaler': options['max_scaler'],
                                       'min_scaler': options['min_scaler'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'augmentation': options.get('augmentation')
                                       }
                        }

        return instructions

    def create(self, source: str, **options):

        img = load_img(source)
        array = np.array(img)

        instructions = {'instructions': array,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        augm_data = None
        if options.get('augmentation') and options.get('augm_data'):
            array, augm_data = self.augmentation_image(image_array=array,
                                                       coords=options['augm_data'],
                                                       augmentation_dict=options['augmentation'])

        frame_mode = options['image_mode'] if 'image_mode' in options.keys() else 'stretch'  # Временное решение

        array = resize_frame(image_array=array,
                             target_shape=(options['height'], options['width']),
                             frame_mode=frame_mode)

        if options['net'] == LayerNetChoice.linear:
            array = array.reshape(np.prod(np.array(array.shape)))
        if options['scaler'] != LayerScalerImageChoice.no_scaler and options.get('preprocess'):
            if options['scaler'] == 'min_max_scaler':
                orig_shape = array.shape
                array = options['preprocess'].transform(array.reshape(-1, 1))
                array = array.reshape(orig_shape).astype('float32')
            elif options['scaler'] == 'terra_image_scaler':
                array = options['preprocess'].transform(array)

        if isinstance(augm_data, str):
            return array, augm_data
        else:
            return array

    @staticmethod
    def augmentation_image(image_array, coords, augmentation_dict):

        # КОСТЫЛЬ ИЗ-ЗА .NATIVE()
        for key, value in augmentation_dict.items():
            if value:
                for name, elem in value.items():
                    if key != 'ChannelShuffle':
                        if isinstance(augmentation_dict[key][name], list):
                            augmentation_dict[key][name] = tuple(augmentation_dict[key][name])
                        elif isinstance(augmentation_dict[key][name], dict):
                            for name2, elem2 in augmentation_dict[key][name].items():
                                augmentation_dict[key][name][name2] = tuple(augmentation_dict[key][name][name2])

        aug_parameters = []
        for key, value in augmentation_dict.items():
            if value:
                aug_parameters.append(getattr(imgaug.augmenters, key)(**value))
        augmentation_object = imgaug.augmenters.Sequential(aug_parameters, random_order=True)

        augmentation_object_data = {'images': np.expand_dims(image_array, axis=0)}
        if coords:
            coords = coords.split(' ')
            for i in range(len(coords)):
                coords[i] = [float(x) for x in coords[i].split(',')]
            bounding_boxes = []
            for elem in coords:
                bounding_boxes.append(imgaug.BoundingBox(*elem))
            bounding_boxes = imgaug.augmentables.bbs.BoundingBoxesOnImage(bounding_boxes,
                                                                          shape=(image_array.shape[0],
                                                                                 image_array.shape[1])
                                                                          )
            augmentation_object_data.update([('bounding_boxes', bounding_boxes)])

        image_array_aug = augmentation_object(**augmentation_object_data)
        if coords:
            bounding_boxes_aug = image_array_aug[1]
            image_array_aug = image_array_aug[0][0]
            bounding_boxes_aug = bounding_boxes_aug.remove_out_of_image().clip_out_of_image()
            aug_coords = []
            for elem in bounding_boxes_aug.bounding_boxes:
                aug_coords.append(
                    ','.join(str(x) for x in [elem.x1_int, elem.y1_int, elem.x2_int, elem.y2_int, elem.label]))
            aug_coords = ' '.join(aug_coords)
            if not aug_coords:
                aug_coords = ''
            return image_array_aug, aug_coords
        else:
            return image_array_aug

    @staticmethod
    def save(sources: list, target: str):
        for elem in sources:
            img_path = elem.split(';')[0]
            os.makedirs(os.path.join(target, os.path.basename(os.path.dirname(img_path))), exist_ok=True)
            shutil.copyfile(img_path, os.path.join(target, os.path.basename(os.path.dirname(img_path)),
                                                   os.path.basename(img_path)))

        paths_list = [os.path.join(target, os.path.basename(os.path.dirname(elem)), os.path.basename(elem))
                      for elem in sources]
        return paths_list
