import os
from collections import OrderedDict
import json
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from terra_ai.cascades.cascade import CascadeElement, CascadeBlock
from terra_ai.general_fucntions.image import change_size
from terra_ai.common import json2model_cascade, ROOT_PATH, load_images
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.common import make_preprocess


INPUT_TRDS_FRONT = load_images(os.path.join(ROOT_PATH, "TerraProjects/airplanes.trds/sources/1_image/Самолеты"))


with open(os.path.join(ROOT_PATH, "test_example/airplanes/config.json")) as cfg:
    config = json.load(cfg)
pprint(config)

data = DatasetData(**config)

pprint(data.inputs.keys())

for img in INPUT_TRDS_FRONT():

    img = img.astype(np.int)
    plt.imshow(img)
    plt.show()


def plot_mask_segmentation(predict, num_classes, classes_colors):
    """
    Returns:
        mask_images
    """

    def _index2color(pix, num_cls, cls_colors):
        index = np.argmax(pix)
        color = []
        for i in range(num_cls):
            if index == i:
                color = cls_colors[i]
        return color

    def _get_colored_mask(mask, num_cls, cls_colors):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        colored_mask = []
        shape_mask = mask.shape
        mask = mask.numpy().reshape(-1, num_cls)

        for pix in range(len(mask)):
            colored_mask.append(
                _index2color(mask[pix], num_cls, cls_colors)
            )
        colored_mask = np.array(colored_mask).astype(np.uint8)

        colored_mask = colored_mask.reshape((shape_mask[0], shape_mask[1], 3))
        return colored_mask

    image = np.squeeze(_get_colored_mask(predict[0], num_classes, classes_colors))

    return image


def plot_mask(num_classes, classes_colors):
    def fun(mask):
        image = plot_mask_segmentation(mask, num_classes=num_classes, classes_colors=classes_colors)
        return image

    return fun


def masked_image(clas=1):
    res = change_size((192, 160))

    def mul_mask(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask
        return np.dstack([r, g, b])

    def fun(img, mask):
        mask = mask[0][:, :, clas]
        mask = np.around(mask)

        img = mul_mask(res(img), mask)
        return img

    return fun


mask2image_cascade = CascadeElement(plot_mask(2, [[0, 0, 0], [255, 0, 0]]), "Изображение маски")

cascade_air = json2model_cascade(os.path.join(ROOT_PATH, "test_example/airplanes/config.json"))

INPUT_TRDS_FRONT = load_images(os.path.join(ROOT_PATH, "TerraProjects/airplanes.trds/sources/1_image/Самолеты"))
mask_crop_cascade = CascadeElement(masked_image(1), name="Вырезание по маске")

example_1 = OrderedDict()
example_1[cascade_air] = ['INPUT']
example_1[mask2image_cascade] = [cascade_air]
example_1 = CascadeBlock(example_1)


example_2 = OrderedDict()
example_2[cascade_air] = ['INPUT']
example_2[mask_crop_cascade] = ['INPUT', cascade_air]
example_2 = CascadeBlock(example_2)


print(example_1)
print(example_2)

# # это делавет фронт
for img in INPUT_TRDS_FRONT():
    plot_img = example_1(img)
    plot_img = plot_img.astype(np.int)

    plt.imshow(plot_img)
    plt.show()

    plot_img = example_2(img)
    plot_img = plot_img.astype(np.int)

    plt.imshow(plot_img)
    plt.show()
