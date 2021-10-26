import os
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.settings import DATASET_ANNOTATION
from terra_ai.datasets.data import AnnotationClassesList
from terra_ai.utils import decamelize

ANNOTATION_SEPARATOR = ":"
ANNOTATION_LABEL_NAME = "# label"
ANNOTATION_COLOR_RGB = "color_rgb"

PATH_TYPE_LIST = [
    decamelize(LayerInputTypeChoice.Image),
    decamelize(LayerOutputTypeChoice.Image),
    decamelize(LayerInputTypeChoice.Audio),
    decamelize(LayerOutputTypeChoice.Audio),
    decamelize(LayerInputTypeChoice.Video),
    decamelize(LayerOutputTypeChoice.ImageSegmentation),
]


def _get_annotation_class(name: str, color: str):
    return {
        "name": name,
        "color": color,
    }


def get_classes_autosearch(
    path: Path, num_classes: int, mask_range: int
) -> AnnotationClassesList:
    def _rgb_in_range(rgb: tuple, target: tuple) -> bool:
        _range0 = range(target[0] - mask_range, target[0] + mask_range)
        _range1 = range(target[1] - mask_range, target[1] + mask_range)
        _range2 = range(target[2] - mask_range, target[2] + mask_range)
        return rgb[0] in _range0 and rgb[1] in _range1 and rgb[2] in _range2

    annotations = AnnotationClassesList()

    for filename in sorted(os.listdir(path)):
        if len(annotations) >= num_classes:
            break

        filepath = Path(path, filename)

        try:
            image = load_img(filepath)
        except Exception:
            continue

        array = img_to_array(image).astype("uint8")
        np_data = array.reshape(-1, 3)
        km = KMeans(n_clusters=num_classes)
        km.fit(np_data)

        cluster_centers = (
            np.round(km.cluster_centers_)
            .astype("uint8")[: max(km.labels_) + 1]
            .tolist()
        )

        for index, rgb in enumerate(cluster_centers, 1):
            if tuple(rgb) in annotations.colors_as_rgb_list:
                continue

            add_condition = True
            for rgb_target in annotations.colors_as_rgb_list:
                if _rgb_in_range(tuple(rgb), rgb_target):
                    add_condition = False
                    break

            if add_condition:
                annotations.append(_get_annotation_class(index, rgb))

    return annotations


def get_classes_annotation(path: Path) -> AnnotationClassesList:
    annotations = AnnotationClassesList()

    try:
        data = pd.read_csv(Path(path, DATASET_ANNOTATION), sep=ANNOTATION_SEPARATOR)
    except FileNotFoundError:
        return annotations

    try:
        for index in range(len(data)):
            annotations.append(
                _get_annotation_class(
                    data.loc[index, ANNOTATION_LABEL_NAME],
                    data.loc[index, ANNOTATION_COLOR_RGB].split(","),
                )
            )
    except KeyError:
        return annotations

    return annotations


def get_yolo_anchors(yolo_version) -> list:

    yolo_anchors: list = []

    if yolo_version == "v3":
        yolo_anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]
    elif yolo_version == "v4":
        yolo_anchors = [
            [[12, 16], [19, 36], [40, 28]],
            [[36, 75], [76, 55], [72, 146]],
            [[142, 110], [192, 243], [459, 401]],
        ]

    return yolo_anchors
