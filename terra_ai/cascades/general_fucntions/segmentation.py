import numpy as np
import tensorflow

from tensorflow.python.framework.ops import EagerTensor


def masked_image(class_id=1):

    def mul_mask(image, mask):
        mask_img = image.copy()

        mask_img[:, :, :, 0] = image[:, :, :, 0] * mask
        mask_img[:, :, :, 1] = image[:, :, :, 1] * mask
        mask_img[:, :, :, 2] = image[:, :, :, 2] * mask

        return mask_img

    def fun(img, mask):

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]

        mask = tensorflow.image.resize(mask, img.shape[1:-1]).numpy()

        mask = mask[:, :, :, class_id]
        mask = np.around(mask)

        img = mul_mask(img, mask)
        return img

    return fun


def plot_mask_segmentation(classes_colors):
    num_class = len(classes_colors)

    def fun(mask, img):

        if len(img.shape) == 3:
            img = img[np.newaxis, ...]

        mask = tensorflow.image.resize(mask, img.shape[1:-1]).numpy()
        img = img.copy()

        for i in range(num_class):
            mask_class = mask[:, :, :, i]
            mask_class = np.around(mask_class)
            mask_class = mask_class.astype(np.bool)

            for j in range(3):
                np.putmask(img[:, :, :, j], mask_class, classes_colors[i][j])

        return img

    return fun
