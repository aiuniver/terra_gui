import os
from collections import OrderedDict

from terra_ai.common import ROOT_PATH, json2cascade, load_images
from terra_ai.cascades.cascade import CascadeElement, CascadeBlock

import tensorflow as tf
import matplotlib.pyplot as plt


# 0 is cat, 1 is dog
cascade_pet = json2cascade(os.path.join(ROOT_PATH, "test_example/petImages/config.json"))
print(cascade_pet)


INPUT_TRDS_FRONT_CAT = load_images(os.path.join(ROOT_PATH, "/home/evgeniy/terra_gui/test_example/petImages/Cat"))
INPUT_TRDS_FRONT_DOG = load_images(os.path.join(ROOT_PATH, "/home/evgeniy/terra_gui/test_example/petImages/Dog"))


def print_classes(score):
    score = score[0]  # только для 1 объекта
    print(
        "На %.2f кот и на %.2f процентов пёс"
        % (100 * (1 - score), 100 * score)
    )


print_cascade = CascadeElement(print_classes, "Показ процентов")

example = OrderedDict()
example[cascade_pet] = ['INPUT']
example[print_cascade] = [cascade_pet]
example = CascadeBlock(example)


# # это делавет фронт
for cat_img, dog_img in zip(INPUT_TRDS_FRONT_CAT(), INPUT_TRDS_FRONT_DOG()):

    example(cat_img)

    plt.imshow(cat_img)
    plt.show()

    example(dog_img)

    plt.imshow(dog_img)
    plt.show()

