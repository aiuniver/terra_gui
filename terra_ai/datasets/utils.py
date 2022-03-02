import os
import cv2
import json
import csv
import tempfile
import zipfile
import numpy as np
import pandas as pd
import xml.etree.ElementTree as Et

from PIL import Image
from ast import literal_eval
from typing import List
from pathlib import Path
from itertools import product
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from terra_ai.data.datasets.extra import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerHandlerChoice,
)
from terra_ai.settings import DATASET_ANNOTATION
from terra_ai.datasets.data import AnnotationClassesList
from terra_ai.utils import decamelize
from terra_ai.data.datasets.extra import LayerODDatasetTypeChoice


ANNOTATION_SEPARATOR = ":"
ANNOTATION_LABEL_NAME = "# label"
ANNOTATION_COLOR_RGB = "color_rgb"

PATH_TYPE_LIST = [
    decamelize(LayerInputTypeChoice.Image),
    decamelize(LayerOutputTypeChoice.Image),
    decamelize(LayerInputTypeChoice.Audio),
    decamelize(LayerOutputTypeChoice.Audio),
    decamelize(LayerInputTypeChoice.Video),
    decamelize(LayerOutputTypeChoice.Segmentation),
]


def _get_annotation_class(name: str, color: str):
    return {
        "name": name,
        "color": color,
    }


def get_classes_autosearch(
    source: Path, path: List[str], num_classes: int, mask_range: int
) -> AnnotationClassesList:
    def _rgb_in_range(rgb: tuple, target: tuple) -> bool:
        _range0 = range(target[0] - mask_range, target[0] + mask_range)
        _range1 = range(target[1] - mask_range, target[1] + mask_range)
        _range2 = range(target[2] - mask_range, target[2] + mask_range)
        return rgb[0] in _range0 and rgb[1] in _range1 and rgb[2] in _range2

    annotations = AnnotationClassesList()

    for dirname in sorted(os.listdir()):
        dirpath = Path(source, dirname)

        for filename in sorted(os.listdir(dirpath)):
            if len(annotations) >= num_classes:
                break

            filepath = Path(dirpath, filename)

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


class Voc:
    """
    Handler Class for VOC PASCAL Format
    """

    @staticmethod
    def generate(data):
        def xml_indent(elem, level=0):
            i = "\n" + level * "\t"
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "\t"
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    xml_indent(elem, level + 1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        xml_list = {}

        for key in data:
            element = data[key]

            xml_annotation = Element("annotation")

            xml_size = Element("size")
            xml_width = Element("width")
            xml_width.text = element["size"]["width"]
            xml_size.append(xml_width)

            xml_height = Element("height")
            xml_height.text = element["size"]["height"]
            xml_size.append(xml_height)

            xml_depth = Element("depth")
            xml_depth.text = element["size"]["depth"]
            xml_size.append(xml_depth)

            xml_annotation.append(xml_size)

            xml_segmented = Element("segmented")
            xml_segmented.text = "0"

            xml_annotation.append(xml_segmented)

            if int(element["objects"]["num_obj"]) < 1:
                return False, "number of Object less than 1"

            for i in range(0, int(element["objects"]["num_obj"])):
                xml_object = Element("object")
                obj_name = Element("name")
                obj_name.text = element["objects"][str(i)]["name"]
                xml_object.append(obj_name)

                obj_pose = Element("pose")
                obj_pose.text = "Unspecified"
                xml_object.append(obj_pose)

                obj_truncated = Element("truncated")
                obj_truncated.text = "0"
                xml_object.append(obj_truncated)

                obj_difficult = Element("difficult")
                obj_difficult.text = "0"
                xml_object.append(obj_difficult)

                xml_bndbox = Element("bndbox")

                obj_xmin = Element("xmin")
                obj_xmin.text = element["objects"][str(i)]["bndbox"]["xmin"]
                xml_bndbox.append(obj_xmin)

                obj_ymin = Element("ymin")
                obj_ymin.text = element["objects"][str(i)]["bndbox"]["ymin"]
                xml_bndbox.append(obj_ymin)

                obj_xmax = Element("xmax")
                obj_xmax.text = element["objects"][str(i)]["bndbox"]["xmax"]
                xml_bndbox.append(obj_xmax)

                obj_ymax = Element("ymax")
                obj_ymax.text = element["objects"][str(i)]["bndbox"]["ymax"]
                xml_bndbox.append(obj_ymax)
                xml_object.append(xml_bndbox)

                xml_annotation.append(xml_object)

            xml_indent(xml_annotation)

            xml_list[key.split(".")[0]] = xml_annotation
        return xml_list

    @staticmethod
    def save(xml_list, path):
        path = os.path.abspath(path)

        for key in xml_list:
            xml = xml_list[key]
            filepath = os.path.join(path, "".join([key, ".xml"]))
            ElementTree(xml).write(filepath)

    @staticmethod
    def parse(paths_list, tmp_lst):
        data = {}
        for filename in paths_list:
            xml = open(filename, "r", encoding="utf-8")
            tree = Et.parse(xml)
            xml.close()
            root = tree.getroot()

            xml_size = root.find("size")
            size = {
                "width": xml_size.find("width").text,
                "height": xml_size.find("height").text,
                "depth": xml_size.find("depth").text,
            }

            objects = root.findall("object")
            if len(objects) == 0:
                return False, "number object zero"

            obj = {"num_obj": len(objects)}

            obj_index = 0
            for _object in objects:
                tmp = {"name": _object.find("name").text}

                xml_bndbox = _object.find("bndbox")
                bndbox = {
                    "xmin": float(xml_bndbox.find("xmin").text),
                    "ymin": float(xml_bndbox.find("ymin").text),
                    "xmax": float(xml_bndbox.find("xmax").text),
                    "ymax": float(xml_bndbox.find("ymax").text),
                }
                tmp["bndbox"] = bndbox
                obj[str(obj_index)] = tmp

                obj_index += 1

            annotation = {"size": size, "objects": obj}

            data[root.find("filename").text.split(".")[0]] = annotation

        return data, {}


class Coco:
    """Handler Class for COCO Format"""

    @staticmethod
    def parse(paths_list, tmp_lst):
        json_path = paths_list[0]
        json_data = json.load(open(json_path))

        images_info = json_data["images"]
        cls_info = json_data["categories"]

        data = {}
        cls_hierarchy = {}

        for anno in json_data["annotations"]:

            image_id = anno["image_id"]
            cls_id = anno["category_id"]

            filename = None
            img_width = None
            img_height = None
            cls = None

            for info in images_info:
                if info["id"] == image_id:
                    filename, img_width, img_height = (
                        info["file_name"].split(".")[0],
                        info["width"],
                        info["height"],
                    )

            for category in cls_info:
                if category["id"] == cls_id:
                    cls = category["name"]
                    cls_parent = (
                        category["supercategory"]
                        if "supercategory" in category
                        else None
                    )

                    if cls not in cls_hierarchy:
                        cls_hierarchy[cls] = cls_parent

            size = {"width": img_width, "height": img_height, "depth": "3"}

            bndbox = {
                "xmin": anno["bbox"][0],
                "ymin": anno["bbox"][1],
                "xmax": anno["bbox"][2] + anno["bbox"][0],
                "ymax": anno["bbox"][3] + anno["bbox"][1],
            }

            obj_info = {"name": cls, "bndbox": bndbox}

            if filename in data:
                obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                data[filename]["objects"][str(obj_idx)] = obj_info
                data[filename]["objects"]["num_obj"] = int(obj_idx) + 1

            elif filename not in data:

                obj = {"num_obj": "1", "0": obj_info}

                data[filename] = {"size": size, "objects": obj}
        return data, cls_hierarchy


class Udacity:
    """
    Handler Class for UDACITY Format
    """

    @staticmethod
    def parse(paths_list, tmp_lst):
        raw_f = open(paths_list[0], "r", encoding="utf-8")
        csv_f = csv.reader(raw_f)
        raw_f.seek(0)
        data = {}

        for line in csv_f:

            raw_line = line[0].split(" ")
            raw_line_length = len(raw_line)

            filename = raw_line[0].split(".")[0]
            xmin = float(raw_line[1])
            ymin = float(raw_line[2])
            xmax = float(raw_line[3])
            ymax = float(raw_line[4])
            cls = raw_line[6].split('"')[1]

            if raw_line_length == 8:
                state = raw_line[7].split('"')[1]
                cls = cls + state

            bndbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}

            obj_info = {"name": cls, "bndbox": bndbox}

            if filename in data:
                obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                data[filename]["objects"][str(obj_idx)] = obj_info
                data[filename]["objects"]["num_obj"] = int(obj_idx) + 1
            elif filename not in data:
                obj = {"num_obj": "1", "0": obj_info}

                data[filename] = {"objects": obj}
        return data, {}


class Kitti:
    """
    Handler Class for KITTI Format
    """

    @staticmethod
    def parse(paths_list, tmp_lst):
        data = {}

        for filename in paths_list:
            txt = open(filename, "r")
            filename = filename.split(".")[0]

            obj = {}
            obj_cnt = 0

            for line in txt:
                elements = line.split(" ")
                name = elements[0]
                if name == "DontCare":
                    continue

                xmin = elements[4]
                ymin = elements[5]
                xmax = elements[6]
                ymax = elements[7]

                bndbox = {
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax),
                }

                obj_info = {"name": name, "bndbox": bndbox}

                obj[str(obj_cnt)] = obj_info
                obj_cnt += 1

            obj["num_obj"] = obj_cnt

            data[filename] = {"objects": obj}
        return data, {}


class Yolo_terra:
    """
    Handler Class for YOLO Format
    """

    def __init__(self, classes, cls_hierarchy={}):

        self.cls_list = classes
        self.cls_hierarchy = cls_hierarchy

    def generate(self, data):
        result = {}

        for key in data:
            contents = ""

            for idx in range(0, int(data[key]["objects"]["num_obj"])):

                xmin = data[key]["objects"][str(idx)]["bndbox"]["xmin"]
                ymin = data[key]["objects"][str(idx)]["bndbox"]["ymin"]
                xmax = data[key]["objects"][str(idx)]["bndbox"]["xmax"]
                ymax = data[key]["objects"][str(idx)]["bndbox"]["ymax"]

                bb = (int(xmin), int(ymin), int(xmax), int(ymax))
                cls_name = data[key]["objects"][str(idx)]["name"]

                def get_class_index(cls_list, cls_hierarchy, cls_name):
                    if cls_name in cls_list:
                        return cls_list.index(cls_name)

                    if type(cls_hierarchy) is dict and cls_name in cls_hierarchy:
                        return get_class_index(
                            cls_list, cls_hierarchy, cls_hierarchy[cls_name]
                        )

                    return None

                cls_id = get_class_index(self.cls_list, self.cls_hierarchy, cls_name)

                bndbox = "".join(["".join([str(e), ","]) for e in bb])
                contents = "".join([contents, bndbox[:-1], ",", str(cls_id), " "])

            result[key] = contents[:-1]
        return result


class Yolov1:
    @staticmethod
    def coordinateCvt2YOLO(size, box):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]

        # (xmin + xmax / 2)
        x = (box[0] + box[1]) / 2.0
        # (ymin + ymax / 2)
        y = (box[2] + box[3]) / 2.0

        # (xmax - xmin) = w
        w = box[1] - box[0]
        # (ymax - ymin) = h
        h = box[3] - box[2]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return round(x, 3), round(y, 3), round(w, 3), round(h, 3)

    @staticmethod
    def parse(paths_list, classes_names_list):
        data = {}
        folder = os.sep.join(paths_list[0].split(os.sep)[:-1])
        for ann_file in paths_list:
            txt = open(ann_file, "r")
            filename = ann_file.split(os.sep)[-1].split(".")[0]
            for im_txt in os.listdir(folder):
                img_name = im_txt.split(".")[0]
                if filename == img_name and not im_txt.endswith(".txt"):
                    img = Image.open(os.path.join(folder, im_txt))
                    break

            img_width = str(img.size[0])
            img_height = str(img.size[1])
            img_depth = 3

            size = {"width": img_width, "height": img_height, "depth": img_depth}

            obj = {}
            obj_cnt = 0

            for line in txt:
                elements = line.split(" ")
                name_id = int(elements[0])

                xminAddxmax = float(elements[1]) * (2.0 * float(img_width))
                yminAddymax = float(elements[2]) * (2.0 * float(img_height))
                w = float(elements[3]) * float(img_width)
                h = float(elements[4]) * float(img_height)

                xmin = (xminAddxmax - w) / 2
                ymin = (yminAddymax - h) / 2
                xmax = xmin + w
                ymax = ymin + h

                bndbox = {
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax),
                }

                obj_info = {
                    "bndbox": bndbox,
                    "name": classes_names_list[name_id],
                }

                obj[str(obj_cnt)] = obj_info
                obj_cnt += 1

            obj["num_obj"] = obj_cnt

            data[filename] = {"size": size, "objects": obj}
        return data, {}


def resize_bboxes(frame_mode, coords, orig_x, orig_y, target_x=416, target_y=416):
    real_boxes = []
    if frame_mode == "stretch":
        for coord in coords.split(" "):
            sample = [literal_eval(x) for x in coord.split(",")]
            sample[0] = int(round((sample[0] / orig_x) * target_x, 0))
            sample[1] = int(round((sample[1] / orig_y) * target_y, 0))
            sample[2] = int(round((sample[2] / orig_x) * target_x, 0))
            sample[3] = int(round((sample[3] / orig_y) * target_y, 0))

            real_boxes.append(sample)

    elif frame_mode == "fit":
        for coord in coords.split(" "):
            sample = [literal_eval(x) for x in coord.split(",")]
            if orig_x >= orig_y:
                new_y = int(orig_y / (orig_x / target_x))
                sample[0] = int(round((sample[0] / orig_x) * target_x, 0))
                sample[2] = int(round((sample[2] / orig_x) * target_x, 0))
                sample[1] = int(
                    round((sample[1] / orig_y) * new_y, 0) + (target_y - new_y) / 2
                )
                sample[3] = int(
                    round((sample[3] / orig_y) * new_y, 0) + (target_y - new_y) / 2
                )
                if new_y > target_y:
                    new_x = int(orig_x / (orig_y / target_y))
                    sample[0] = int(
                        round((sample[0] / orig_x) * new_x, 0) + (target_x - new_x) / 2
                    )
                    sample[2] = int(
                        round((sample[2] / orig_x) * new_x, 0) + (target_x - new_x) / 2
                    )
                    sample[1] = int(round((sample[1] / orig_y) * target_y, 0))
                    sample[3] = int(round((sample[3] / orig_y) * target_y, 0))

            elif orig_y >= orig_x:
                new_x = int(orig_x / (orig_y / target_y))
                sample[0] = int(
                    round((sample[0] / orig_x) * new_x, 0) + (target_x - new_x) / 2
                )
                sample[2] = int(
                    round((sample[2] / orig_x) * new_x, 0) + (target_x - new_x) / 2
                )
                sample[1] = int(round((sample[1] / orig_y) * target_y, 0))
                sample[3] = int(round((sample[3] / orig_y) * target_y, 0))
                if new_x > target_x:
                    new_y = int(orig_y / (orig_x / target_x))
                    sample[0] = int(round((sample[0] / orig_x) * target_x, 0))
                    sample[2] = int(round((sample[2] / orig_x) * target_x, 0))
                    sample[1] = int(
                        round((sample[1] / orig_y) * new_y, 0) + (target_y - new_y) / 2
                    )
                    sample[3] = int(
                        round((sample[3] / orig_y) * new_y, 0) + (target_y - new_y) / 2
                    )

            real_boxes.append(sample)

    elif frame_mode == "cut":
        for coord in coords.split(" "):
            sample = [literal_eval(x) for x in coord.split(",")]
            if orig_x <= target_x:
                sample[0] = int(sample[0] + (target_x - orig_x) / 2)
                sample[2] = int(sample[2] + (target_x - orig_x) / 2)
            else:
                sample[0] = int(sample[0] - (orig_x - target_x) / 2)
                sample[2] = int(sample[2] - (orig_x - target_x) / 2)
            if orig_y <= target_y:
                sample[1] = int(sample[1] + (target_y - orig_y) / 2)
                sample[3] = int(sample[3] + (target_y - orig_y) / 2)
            else:
                sample[1] = int(sample[1] - (orig_y - target_y) / 2)
                sample[3] = int(sample[3] - (orig_y - target_y) / 2)

            real_boxes.append(sample)

    pop_idxs = []
    for idx, bbox in enumerate(real_boxes):
        for i in range(4):
            if i in [0, 2]:
                if bbox[i] > target_x:
                    bbox[i] = target_x
            elif i in [1, 3]:
                if bbox[i] > target_y:
                    bbox[i] = target_y
            if bbox[i] < 0:
                bbox[i] = 0

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            pop_idxs.append(idx)

    for i in reversed(pop_idxs):
        real_boxes.pop(i)

    return real_boxes


# def resize_bboxes(coords, orig_x, orig_y, target_x=416, target_y=416):
#
#     real_boxes = []
#     for coord in coords.split(' '):
#         sample = [literal_eval(x) for x in coord.split(',')]
#         sample[0] = int(round((sample[0] / orig_x) * target_x, 0))
#         sample[1] = int(round((sample[1] / orig_y) * target_y, 0))
#         sample[2] = int(round((sample[2] / orig_x) * target_x, 0))
#         sample[3] = int(round((sample[3] / orig_y) * target_y, 0))
#
#         if sample[0] < sample[2] and sample[1] < sample[3]:
#             real_boxes.append(sample)
#
#     return real_boxes


def get_image_size(path):

    img = Image.open(path)

    return img.height, img.width


def get_od_names(version_data, source_path, version_path_data):

    names_list = []
    for handler in version_data.outputs:
        if handler.type == "handler" and handler.parameters.type in [
            LayerHandlerChoice.YoloV3,
            LayerHandlerChoice.YoloV4,
        ]:
            ann_path = version_data.outputs.get(handler.bind.up[0]).parameters.data[0]
            if handler.parameters.options.model_type in [
                LayerODDatasetTypeChoice.Yolov1,
                LayerODDatasetTypeChoice.Yolo_terra,
            ]:
                with open(Path(source_path).joinpath("obj.names"), "r") as names:
                    names_list = names.read()
                names_list = [elem for elem in names_list.split("\n") if elem]

            elif handler.parameters.options.model_type == LayerODDatasetTypeChoice.Coco:
                for js_file in os.listdir(
                    os.path.join(version_data.version_path_data, ann_path)
                ):
                    json_data = json.load(
                        open(
                            os.path.join(
                                version_data.version_path_data, ann_path, js_file
                            )
                        )
                    )

                names_list = [0 for i in json_data["categories"]]
                for i in json_data["categories"]:
                    names_list[i["id"]] = i["name"]

            elif handler.parameters.options.model_type == LayerODDatasetTypeChoice.Voc:
                (dir_path, dir_names, filenames) = next(
                    os.walk(
                        os.path.abspath(
                            os.path.join(version_data.version_path_data, ann_path)
                        )
                    )
                )
                for filename in filenames:
                    xml = open(os.path.join(dir_path, filename), "r", encoding="utf-8")
                    tree = Et.parse(xml)
                    root = tree.getroot()
                    objects = root.findall("object")
                    for _object in objects:
                        names_list.append(_object.find("name").text)
                    xml.close()
                names_list = sorted(set(names_list))

            elif (
                handler.parameters.options.model_type == LayerODDatasetTypeChoice.Kitti
            ):
                (dir_path, dir_names, filenames) = next(
                    os.walk(
                        os.path.abspath(
                            os.path.join(version_data.version_path_data, ann_path)
                        )
                    )
                )
                for filename in filenames:
                    txt = open(os.path.join(dir_path, filename), "r")
                    for line in txt:
                        elements = line.split(" ")
                        names_list.append(elements[0])
                    txt.close()
                names_list = sorted(set(names_list))

            elif (
                handler.parameters.options.model_type
                == LayerODDatasetTypeChoice.Udacity
            ):
                for i in os.listdir(version_path_data):
                    if i.endswith(".csv"):
                        raw_f = open(
                            os.path.join(version_data.version_path_data, i),
                            "r",
                            encoding="utf-8",
                        )
                csv_f = csv.reader(raw_f)
                raw_f.seek(0)

                for line in csv_f:
                    raw_line = line[0].split(" ")
                    raw_line_length = len(raw_line)
                    cls = raw_line[6].split('"')[1]
                    if raw_line_length == 8:
                        state = raw_line[7].split('"')[1]
                        cls = cls + state
                    names_list.append(cls)
                names_list = sorted(set(names_list))

    return names_list


def get_annotation_type_autosearch(path: Path) -> LayerODDatasetTypeChoice:
    dir_names = []
    file_names = []

    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)):
            dir_names.append(filename)
        elif filename.endswith(".csv"):
            return LayerODDatasetTypeChoice.Udacity
        else:
            file_names.append(filename)

    if len(dir_names) == 1:
        return LayerODDatasetTypeChoice.Yolov1

    for dir_name in dir_names:
        if os.listdir(os.path.join(path, dir_name))[0].endswith(".json"):
            return LayerODDatasetTypeChoice.Coco
        elif os.listdir(os.path.join(path, dir_name))[0].endswith(".xml"):
            return LayerODDatasetTypeChoice.Voc
        elif (
            os.listdir(os.path.join(path, dir_name))[0].endswith(".txt")
            and "obj.names" not in file_names
        ):
            return LayerODDatasetTypeChoice.Kitti
        elif (
            os.listdir(os.path.join(path, dir_name))[0].endswith(".txt")
            and "obj.names" in file_names
        ):
            return LayerODDatasetTypeChoice.Yolo_terra
        else:
            annotation_type = "Не определено"

    return annotation_type


def resize_frame(image_array, target_shape, frame_mode):
    original_shape = (image_array.shape[0], image_array.shape[1])
    resized = None
    if frame_mode == "stretch":
        resized = cv2.resize(image_array, (target_shape[1], target_shape[0]))

    elif frame_mode == "fit":
        if image_array.shape[1] >= image_array.shape[0]:
            resized_shape = list(target_shape).copy()
            resized_shape[0] = int(
                image_array.shape[0] / (image_array.shape[1] / target_shape[1])
            )
            if resized_shape[0] > target_shape[0]:
                resized_shape = list(target_shape).copy()
                resized_shape[1] = int(
                    image_array.shape[1] / (image_array.shape[0] / target_shape[0])
                )
            image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
        elif image_array.shape[0] >= image_array.shape[1]:
            resized_shape = list(target_shape).copy()
            resized_shape[1] = int(
                image_array.shape[1] / (image_array.shape[0] / target_shape[0])
            )
            if resized_shape[1] > target_shape[1]:
                resized_shape = list(target_shape).copy()
                resized_shape[0] = int(
                    image_array.shape[0] / (image_array.shape[1] / target_shape[1])
                )
            image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
        resized = image_array
        if resized.shape[0] < target_shape[0]:
            black_bar = np.zeros(
                (int((target_shape[0] - resized.shape[0]) / 2), resized.shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized))
            black_bar_2 = np.zeros(
                (int((target_shape[0] - resized.shape[0])), resized.shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2))
        if resized.shape[1] < target_shape[1]:
            black_bar = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1]) / 2), 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized), axis=1)
            black_bar_2 = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2), axis=1)

    elif frame_mode == "cut":
        resized = image_array.copy()
        if original_shape[0] > target_shape[0]:
            resized = resized[
                int(original_shape[0] / 2 - target_shape[0] / 2) : int(
                    original_shape[0] / 2 - target_shape[0] / 2
                )
                + target_shape[0],
                :,
            ]
        else:
            black_bar = np.zeros(
                (int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized))
            black_bar_2 = np.zeros(
                (int((target_shape[0] - resized.shape[0])), original_shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2))
        if original_shape[1] > target_shape[1]:
            resized = resized[
                :,
                int(original_shape[1] / 2 - target_shape[1] / 2) : int(
                    original_shape[1] / 2 - target_shape[1] / 2
                )
                + target_shape[1],
            ]
        else:
            black_bar = np.zeros(
                (target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized), axis=1)
            black_bar_2 = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2), axis=1)
    return resized


def zip_dataset(src, dst):
    zf = zipfile.ZipFile("%s.zip" % (dst), "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dirname, subdirs, files in os.walk(src):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src) + 1 :]
            zf.write(absname, arcname)
    zf.close()


def make_tracker_dataset(source_path, dst_path, bboxes, frame_mode):
    def make_crop(image_path, bbox):
        image = cv2.imread(image_path)
        coor = np.array(bbox[:4], dtype=np.int32)
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        crop = image[y1:y2, x1:x2]
        return crop

    img_path = os.path.join(source_path, "Images")
    tmp_directory = tempfile.mkdtemp()
    ims1 = []
    ims2 = []
    classes = []
    drop_idxs = []
    border = 30
    height = 0
    width = 0
    idx = 0
    while idx < len(bboxes) - 1:
        anns1 = bboxes[idx]
        anns2 = bboxes[idx + 1]
        os.makedirs(os.path.join(tmp_directory, f"frame_{idx}"), exist_ok=True)
        for j, ann in enumerate(anns1):
            crop = make_crop(
                os.path.join(img_path, sorted(os.listdir(img_path))[idx]), ann
            )
            cv2.imwrite(
                os.path.join(tmp_directory, f"frame_{idx}", f"crop_{j}.jpeg"), crop
            )
            if crop.shape[0] > height:
                height = crop.shape[0]
            if crop.shape[1] > width:
                width = crop.shape[1]

        for a1, a2 in product(anns1, anns2):
            if (
                abs(a1[0] - a2[0]) <= border
                and abs(a1[1] - a2[1]) <= border
                and a1[4] == a2[4]
            ):
                classes.append("Одинаковые")
            else:
                classes.append("Разные")
            ims1.append(os.path.join(f"frame_{idx}", f"crop_{anns1.index(a1)}.jpeg"))
            ims2.append(
                os.path.join(f"frame_{idx + 1}", f"crop_{anns2.index(a2)}.jpeg")
            )
        idx += 1

    for j, ann in enumerate(bboxes[-1]):
        crop = make_crop(os.path.join(img_path, sorted(os.listdir(img_path))[-1]), ann)
        cv2.imwrite(
            os.path.join(tmp_directory, f"frame_{len(bboxes)}", f"crop_{j}.jpeg"), crop
        )
        if crop.shape[0] > height:
            height = crop.shape[0]
        if crop.shape[1] > width:
            width = crop.shape[1]

    for directory in os.listdir(tmp_directory):
        for im_name in os.listdir(os.path.join(tmp_directory, directory)):
            img = cv2.imread(
                os.path.join(tmp_directory, directory, im_name), cv2.IMREAD_UNCHANGED
            )
            resized_im = resize_frame(img, (height, width), frame_mode)
            cv2.imwrite(os.path.join(tmp_directory, directory, im_name), resized_im)

    tracker_table = pd.DataFrame({"img_1": ims1, "img_2": ims2, "class": classes})

    crops_list = sorted(set(tracker_table["img_1"].tolist()))
    for crop in crops_list:
        tmp_df = tracker_table[tracker_table["img_1"] == crop]
        if len(tmp_df[tmp_df["class"] == "Одинаковые"]) > 1:
            drop_idxs.append(tmp_df[tmp_df["class"] == "Одинаковые"].index.tolist())
    for idxs in drop_idxs:
        tracker_table.drop(index=idxs, inplace=True)
    tracker_table.index = range(0, len(tracker_table))

    tracker_table.to_csv(os.path.join(tmp_directory, "tracker.csv"), index=False)
    zip_dataset(tmp_directory, os.path.join(dst_path, "tracker"))
