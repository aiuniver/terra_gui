from PIL import Image
from sklearn.cluster import KMeans
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from pathlib import Path
from terra_ai.settings import DATASET_ANNOTATION

ANNOTATION_SEPARATOR = ":"
ANNOTATION_LABEL_NAME = "# label"
ANNOTATION_COLOR_RGB = "color_rgb"


def get_classes_autosearch(folder_path: str, num_classes: int, mask_range: int) -> dict:
    color_dict = {}
    idx = 1
    for img in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, img)
        width, height = Image.open(path).size
        img = load_img(path, target_size=(height, width))
        array = img_to_array(img).astype("uint8")

        image = array.reshape(-1, 3)
        km = KMeans(n_clusters=num_classes)
        km.fit(image)
        labels = km.labels_
        cl_cent = (
            np.round(km.cluster_centers_).astype("uint8")[: max(labels) + 1].tolist()
        )
        add_condition = False

        for color in cl_cent:
            if color_dict:
                if color not in color_dict.values():
                    for in_color in color_dict.values():
                        if (
                            color[0]
                            in range(in_color[0] - mask_range, in_color[0] + mask_range)
                            and color[1]
                            in range(in_color[1] - mask_range, in_color[1] + mask_range)
                            and color[2]
                            in range(in_color[2] - mask_range, in_color[2] + mask_range)
                        ):
                            add_condition = False
                            break
                        else:
                            add_condition = True
                    if add_condition:
                        color_dict[str(idx)] = color
                        idx += 1
            else:
                color_dict[str(idx)] = color
                idx += 1
        if len(color_dict) >= num_classes:
            break

    return color_dict


def get_classes_annotation(dataset_path):
    txt = pd.read_csv(Path(dataset_path, DATASET_ANNOTATION), sep=ANNOTATION_SEPARATOR)
    color_dict = {}
    for i in range(len(txt)):
        color_dict[txt.loc[i, ANNOTATION_LABEL_NAME]] = [
            int(num) for num in txt.loc[i, ANNOTATION_COLOR_RGB].split(",")
        ]

    return color_dict
