import os
import json
import random
import shutil

from PIL import Image
from typing import List, Optional
from pathlib import Path

from terra_ai import settings
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.deploy.extra import TaskTypeChoice
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.training.guinn import interactive


class BaseCollectionList(List):
    _path: Optional[Path]

    def __init__(self, *args, path: Path = None, **kwargs):
        self._path = path
        if self._path is not None:
            os.makedirs(self._path, exist_ok=True)
        super().__init__(*args, **kwargs)
        self._reset()

    def _reset(self):
        self.clear()
        for _ in list(range(settings.DEPLOY_PRESET_COUNT)):
            self.append(None)
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)
            os.makedirs(self._path, exist_ok=True)

    def try_init(self):
        if not list(filter(None, self)) and interactive.deploy_presets_data:
            self.reload(list(range(settings.DEPLOY_PRESET_COUNT)))

    def reload(self, range_indexes: List):
        raise MethodNotImplementedException("reload", self.__class__.__name__)


class ImageClassification(BaseCollectionList):
    def reload(self, range_indexes: List):
        source = interactive.deploy_presets_data
        label_file = Path(self._path, "label.txt")
        label = []
        if not source:
            self._reset()
            try:
                os.remove(label_file)
            except Exception:
                pass
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            filepath = Path(value.get("source"))
            destination = Path(self._path, f"{index + 1}.jpg")
            filepath_im = Image.open(filepath)
            filepath_im.save(destination)
            value.update({"source": str(destination.absolute())})
            self[index] = value

        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(label_file, "w") as labelfile_ref:
            labelfile_ref.write("\n".join(label))


class ImageSegmentation(BaseCollectionList):
    def reload(self, range_indexes: List):
        source_path = Path(self._path, "preset", "in")
        segment_path = Path(self._path, "preset", "out")
        os.makedirs(source_path, exist_ok=True)
        os.makedirs(segment_path, exist_ok=True)
        source = interactive.deploy_presets_data
        labelfile = Path(self._path, "label.txt")
        label = []
        if not source:
            self._reset()
            try:
                os.remove(labelfile)
            except Exception:
                pass
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
                os.remove(self[index].get("segment"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            filepath_source = Path(value.get("source"))
            filepath_segment = Path(value.get("segment"))
            destination_source = Path(source_path, f"{index + 1}.jpg")
            destination_segment = Path(segment_path, f"{index + 1}.jpg")
            filepath_source_im = Image.open(filepath_source)
            filepath_source_im.save(destination_source)
            filepath_segment_im = Image.open(filepath_segment)
            filepath_segment_im.save(destination_segment)
            value.update(
                {
                    "source": str(destination_source.absolute()),
                    "segment": str(destination_segment.absolute()),
                }
            )
            self[index] = value

        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(labelfile, "w") as labelfile_ref:
            labelfile_ref.write("\n".join(label))


class TextClassification(BaseCollectionList):
    def reload(self, range_indexes: List):
        source_path = Path(self._path, "preset", "in")
        predict_path = Path(self._path, "preset", "out")
        os.makedirs(source_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)
        source = interactive.deploy_presets_data
        predict_file = Path(predict_path, "predict.txt")
        label = []
        if not source:
            self._reset()
            try:
                os.remove(predict_file)
            except Exception:
                pass
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            destination_source = Path(source_path, f"{index + 1}.txt")
            with open(destination_source, "w") as destination_source_ref:
                destination_source_ref.write(
                    json.dumps(value.get("source", ""), ensure_ascii=False)
                )
            self[index] = value

        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(predict_file, "w") as predictfile_ref:
            predictfile_ref.write("\n".join(label))


class TextTextSegmentation(BaseCollectionList):
    def reload(self, range_indexes: List):
        source_path = Path(self._path, "preset", "in")
        format_path = Path(self._path, "preset", "out")
        os.makedirs(source_path, exist_ok=True)
        os.makedirs(format_path, exist_ok=True)
        source = interactive.deploy_presets_data
        if not source:
            self._reset()
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
                os.remove(self[index].get("format"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            destination_source = Path(source_path, f"{index + 1}.txt")
            destination_format = Path(format_path, f"{index + 1}.txt")
            with open(destination_source, "w") as destination_source_ref:
                destination_source_ref.write(value.get("source", ""))
            with open(destination_format, "w") as destination_format_ref:
                destination_format_ref.write(value.get("format", ""))
            self[index] = value


class VideoClassification(BaseCollectionList):
    def reload(self, range_indexes: List):
        source_path = Path(self._path, "preset", "in")
        predict_path = Path(self._path, "preset", "out")
        os.makedirs(source_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)
        source = interactive.deploy_presets_data
        label_file = Path(predict_path, "label.txt")
        label = []

        if not source:
            self._reset()
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            destination_source = Path(source_path, f"{index + 1}.txt")
            with open(destination_source, "w") as destination_source_ref:
                destination_source_ref.write(value.get("source", ""))
            self[index] = value

        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(label_file, "w") as label_file_ref:
            label_file_ref.write("\n".join(label))


class AudioClassification(BaseCollectionList):
    def reload(self, range_indexes: List):
        source = interactive.deploy_presets_data
        label = []

        if not source:
            self._reset()
            return

        for index in range_indexes:
            try:
                os.remove(self[index].get("source"))
            except Exception:
                pass
            value = dict(source[random.randint(0, len(source) - 1)])
            audio = Path(value.get("source"))
            shutil.copyfile(audio, Path(self._path, f"{index + 1}{audio.suffix}"))
            self[index] = value

        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(Path(self._path, "label.txt"), "w") as label_file_ref:
            label_file_ref.write("\n".join(label))


class TableDataClassification(BaseCollectionList):
    def reload(self, range_indexes: List):
        source_path = Path(self._path)
        predict_path = Path(self._path, "preset", "out")
        os.makedirs(source_path, exist_ok=True)
        os.makedirs(predict_path, exist_ok=True)
        source = interactive.deploy_presets_data

        preset_file = Path(self._path, "preset.txt")
        label_file = Path(self._path, "label.txt")

        if not source:
            self._reset()
            return

        for index in range_indexes:
            value = dict(source[random.randint(0, len(source) - 1)])
            with open(preset_file, "a") as preset_file_ref:
                preset_file_ref.write(json.dumps(value.get("source", ""), ensure_ascii=False))
                preset_file_ref.write('\n')
            self[index] = value

        label = []
        for item in self:
            label.append(json.dumps(item.get("data", []), ensure_ascii=False))
        with open(label_file, "a") as label_file_ref:
            label_file_ref.write("\n".join(label))


class TableDataRegression(TableDataClassification):
    def reload(self, range_indexes: List):

        preset_file = Path(self._path, "preset.txt")
        label_file = Path(self._path, "label.txt")
        source = interactive.deploy_presets_data

        if not source:
            self._reset()
            return

        for index in range_indexes:
            random_index = random.randint(0, len(source["label"]) - 1)
            value = {
                "label": source["label"][random_index],
                "preset": source["preset"][random_index],
            }
            with open(preset_file, "a") as preset_file_ref:
                preset_file_ref.write(json.dumps(value["preset"], ensure_ascii=False))
                preset_file_ref.write("\n")
            with open(label_file, "a") as label_file_ref:
                label_file_ref.write(json.dumps(value["label"], ensure_ascii=False))
                label_file_ref.write("\n")
            self[index] = value


class BaseCollection(BaseMixinData):
    type: TaskTypeChoice
    data: Optional[BaseCollectionList]

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        data.update({"data": self.data if len(list(filter(None, self.data))) else None})
        return data
