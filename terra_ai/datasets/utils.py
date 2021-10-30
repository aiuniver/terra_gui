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
from terra_ai.data.datasets.creation import CreationData

import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
import csv
from PIL import Image
from ast import literal_eval
import json

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

        cluster_centers = (np.round(km.cluster_centers_).astype("uint8")[: max(km.labels_) + 1].tolist())

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

    def xml_indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def generate(self, data):
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
                obj_xmin.text = element["objects"][str(
                    i)]["bndbox"]["xmin"]
                xml_bndbox.append(obj_xmin)

                obj_ymin = Element("ymin")
                obj_ymin.text = element["objects"][str(
                    i)]["bndbox"]["ymin"]
                xml_bndbox.append(obj_ymin)

                obj_xmax = Element("xmax")
                obj_xmax.text = element["objects"][str(
                    i)]["bndbox"]["xmax"]
                xml_bndbox.append(obj_xmax)

                obj_ymax = Element("ymax")
                obj_ymax.text = element["objects"][str(
                    i)]["bndbox"]["ymax"]
                xml_bndbox.append(obj_ymax)
                xml_object.append(xml_bndbox)

                xml_annotation.append(xml_object)

            self.xml_indent(xml_annotation)

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
    def parse(path, path2):
        (dir_path, dir_names, filenames) = next(
            os.walk(os.path.abspath(path)))

        data = {}

        for filename in filenames:

            xml = open(os.path.join(dir_path, filename), "r")

            tree = Et.parse(xml)
            root = tree.getroot()

            xml_size = root.find("size")
            size = {
                "width": xml_size.find("width").text,
                "height": xml_size.find("height").text,
                "depth": xml_size.find("depth").text

            }

            objects = root.findall("object")
            if len(objects) == 0:
                return False, "number object zero"

            obj = {
                "num_obj": len(objects)
            }

            obj_index = 0
            for _object in objects:
                tmp = {
                    "name": _object.find("name").text
                }

                xml_bndbox = _object.find("bndbox")
                bndbox = {
                    "xmin": float(xml_bndbox.find("xmin").text),
                    "ymin": float(xml_bndbox.find("ymin").text),
                    "xmax": float(xml_bndbox.find("xmax").text),
                    "ymax": float(xml_bndbox.find("ymax").text)
                }
                tmp["bndbox"] = bndbox
                obj[str(obj_index)] = tmp

                obj_index += 1

            annotation = {
                "size": size,
                "objects": obj
            }

            data[root.find("filename").text.split(".")[0]] = annotation
        return data, {}


class Coco:
    """
    Handler Class for COCO Format
    """

    @staticmethod
    def parse(json_path, img_path):
        json_data = json.load(open(os.path.join(json_path, 'common.json')))

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
                    filename, img_width, img_height = \
                        info["file_name"].split(
                            ".")[0], info["width"], info["height"]

                    if img_width == 0 or img_height == 0:
                        img = Image.open(os.path.join(
                            img_path, info["file_name"]))
                        img_width = str(img.size[0])
                        img_height = str(img.size[1])

            for category in cls_info:
                if category["id"] == cls_id:
                    cls = category["name"]
                    cls_parent = category['supercategory'] if 'supercategory' in category else None

                    if cls not in cls_hierarchy:
                        cls_hierarchy[cls] = cls_parent

            size = {
                "width": img_width,
                "height": img_height,
                "depth": "3"
            }

            bndbox = {
                "xmin": anno["bbox"][0],
                "ymin": anno["bbox"][1],
                "xmax": anno["bbox"][2] + anno["bbox"][0],
                "ymax": anno["bbox"][3] + anno["bbox"][1]
            }

            obj_info = {
                "name": cls,
                "bndbox": bndbox
            }

            if filename in data:
                obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                data[filename]["objects"][str(obj_idx)] = obj_info
                data[filename]["objects"]["num_obj"] = int(obj_idx) + 1

            elif filename not in data:

                obj = {
                    "num_obj": "1",
                    "0": obj_info
                }

                data[filename] = {
                    "size": size,
                    "objects": obj
                }
        return data, cls_hierarchy


class Udacity:
    """
    Handler Class for UDACITY Format
    """

    @staticmethod
    def parse(csv_path, img_path):
        raw_f = open(os.path.join(csv_path, 'common.csv'), 'r', encoding='utf-8')
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

            img = Image.open(os.path.join(
                img_path, "".join([filename, ".jpg"])))
            img_width = str(img.size[0])
            img_height = str(img.size[1])
            img_depth = 3

            size = {
                "width": img_width,
                "height": img_height,
                "depth": img_depth
            }

            bndbox = {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }

            obj_info = {
                "name": cls,
                "bndbox": bndbox
            }

            if filename in data:
                obj_idx = str(int(data[filename]["objects"]["num_obj"]))
                data[filename]["objects"][str(obj_idx)] = obj_info
                data[filename]["objects"]["num_obj"] = int(obj_idx) + 1
            elif filename not in data:
                obj = {
                    "num_obj": "1",
                    "0": obj_info
                }

                data[filename] = {
                    "size": size,
                    "objects": obj
                }
        return data, {}


class Kitti:
    """
    Handler Class for KITTI Format
    """

    @staticmethod
    def parse(label_path, img_path):
        img_type = ".jpg"
        # with open("box_groups.txt", "w") as bboxGroups:
        (dir_path, dir_names, filenames) = next(
            os.walk(os.path.abspath(label_path)))

        data = {}

        for filename in filenames:
            txt = open(os.path.join(dir_path, filename), "r")
            filename = filename.split(".")[0]

            img = Image.open(os.path.join(
                img_path, "".join([filename, img_type])))
            img_width = str(img.size[0])
            img_height = str(img.size[1])
            img_depth = 3

            size = {
                "width": img_width,
                "height": img_height,
                "depth": img_depth
            }

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
                    "ymax": float(ymax)
                }

                # bboxGroups.write("{} {} {} {}\n".format(float(xmin), float(
                #     ymin), float(xmax) - float(xmin), float(ymax) - float(ymin)))

                obj_info = {
                    "name": name,
                    "bndbox": bndbox
                }

                obj[str(obj_cnt)] = obj_info
                obj_cnt += 1

            obj["num_obj"] = obj_cnt

            data[filename] = {
                "size": size,
                "objects": obj
            }
        return data, {}


class Yolo:
    """
    Handler Class for YOLO Format
    """

    def __init__(self, cls_list_path, cls_hierarchy={}):
        with open(cls_list_path, 'r') as file:
            l = file.read().splitlines()

        self.cls_list = l
        self.cls_hierarchy = cls_hierarchy

    def coordinateCvt2YOLO(self, size, box):

        # (xmin + xmax / 2)
        x = (box[0] + box[1]) / 2.0
        # (ymin + ymax / 2)
        y = (box[2] + box[3]) / 2.0

        # (xmax - xmin) = w
        w = box[1] - box[0]
        # (ymax - ymin) = h
        h = box[3] - box[2]

        return int(x), int(y), int(w), int(h)

    def parse(self, label_path, img_path):
        img_type = ".jpg"
        (dir_path, dir_names, filenames) = next(
            os.walk(os.path.abspath(label_path)))

        data = {}

        for filename in filenames:

            txt = open(os.path.join(dir_path, filename), "r")

            filename = filename.split(".")[0]

            img = Image.open(os.path.join(
                img_path, "".join([filename, img_type])))
            img_width = str(img.size[0])
            img_height = str(img.size[1])
            img_depth = 3

            size = {
                "width": img_width,
                "height": img_height,
                "depth": img_depth
            }

            obj = {}
            obj_cnt = 0

            for line in txt:
                elements = line.split(" ")
                name_id = elements[0]

                xminAddxmax = float(elements[1]) * (2.0 * float(img_width))
                yminAddymax = float(
                    elements[2]) * (2.0 * float(img_height))

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
                    "ymax": float(ymax)
                }

                obj_info = {
                    "bndbox": bndbox,
                    "name": name_id,
                }

                obj[str(obj_cnt)] = obj_info
                obj_cnt += 1

            obj["num_obj"] = obj_cnt

            data[filename] = {
                "size": size,
                "objects": obj
            }
        return data, {}

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
                        return get_class_index(cls_list, cls_hierarchy, cls_hierarchy[cls_name])

                    return None

                cls_id = get_class_index(
                    self.cls_list, self.cls_hierarchy, cls_name)

                bndbox = "".join(["".join([str(e), ","]) for e in bb])
                contents = "".join(
                    [contents, bndbox[:-1], ",", str(cls_id), "\n"])

            result[key] = contents
        return result

    def save(self, data, save_path, img_path, manifest_path):
        if os.path.isdir(manifest_path):
            manifest_abspath = os.path.join(manifest_path, "manifest.txt")
        else:
            manifest_abspath = manifest_path

        with open(os.path.abspath(manifest_abspath), "w") as manifest_file:
            for key in data:
                manifest_file.write(os.path.abspath(os.path.join(
                    img_path, "".join([key, '.jpg', "\n"]))))

                with open(os.path.abspath(os.path.join(save_path, "".join([key, ".txt"]))), "w") as output_txt_file:
                    output_txt_file.write(data[key])


def convert_object_detection(creation_data: CreationData):
    id_sum = len(creation_data.inputs) + len(creation_data.outputs)
    model_type = eval(f'{creation_data.outputs.get(id_sum).parameters.model_type}()')
    data, cls_hierarchy = model_type.parse(creation_data.outputs.get(id_sum).parameters.sources_paths[0],
                                           creation_data.inputs.get(len(creation_data.inputs)).parameters.sources_paths[
                                               0])
    yolo = Yolo(os.path.abspath(creation_data.source_path.joinpath('obj.names')), cls_hierarchy=cls_hierarchy)
    data = yolo.generate(data)
    os.makedirs(os.path.join(creation_data.source_path, 'Yolo_annotations'), exist_ok=True)
    yolo.save(data, Path(os.path.join(creation_data.source_path, 'Yolo_annotations')),
              creation_data.inputs.get(len(creation_data.inputs)).parameters.sources_paths[0],
              creation_data.source_path)


def resize_bboxes(coords, orig_x, orig_y):
    x_scale = orig_x / 416
    y_scale = orig_y / 416
    real_boxes = []
    if x_scale == 1 and y_scale == 1:
        for coord in coords.split(' '):
            real_boxes.append([literal_eval(num) for num in coord.split(',')])
    else:
        for coord in coords.split(' '):
            tmp = []
            for i, num in enumerate(coord.split(',')):
                if i in [0, 2]:
                    tmp_value = int(literal_eval(num) / x_scale) - 1
                    scale_value = orig_x if tmp_value > orig_x else tmp_value
                    tmp.append(scale_value)
                elif i in [1, 3]:
                    tmp_value = int(literal_eval(num) / y_scale) - 1
                    scale_value = orig_y if tmp_value > orig_y else tmp_value
                    tmp.append(scale_value)
                else:
                    tmp.append(literal_eval(num))
            real_boxes.append(tmp)
    return real_boxes
