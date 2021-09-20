import os
from collections import OrderedDict

from terra_ai.common import ROOT_PATH, json2model_cascade, load_images
from terra_ai.cascades.cascade import CascadeElement, CascadeBlock

import numpy as np
import matplotlib.pyplot as plt


# 0 is cat, 1 is dog
cascade_pet = json2model_cascade(os.path.join(ROOT_PATH, "test_example/petImages/config.json"))
print(cascade_pet)


INPUT_TRDS_FRONT_CAT = load_images(os.path.join(ROOT_PATH, os.path.join(ROOT_PATH, "test_example/petImages/Cat")))
INPUT_TRDS_FRONT_DOG = load_images(os.path.join(ROOT_PATH, os.path.join(ROOT_PATH, "test_example/petImages/Dog")))


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

    cat_img = cat_img.astype(np.int)
    plt.imshow(cat_img)
    plt.show()

    example(dog_img)

    dog_img = dog_img.astype(np.int)
    plt.imshow(dog_img)
    plt.show()
