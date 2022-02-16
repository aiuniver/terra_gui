import json
import os
import numpy as np

from typing import Any

from .arrays_classes.image import ImageArray
from .arrays_classes.audio import AudioArray
from .arrays_classes.classification import ClassificationArray
from .arrays_classes.text import TextArray
from .arrays_classes.video import VideoArray
from .arrays_classes.discriminator import DiscriminatorArray
from .arrays_classes.generator import GeneratorArray
from .arrays_classes.noise import NoiseArray
from .arrays_classes.object_detection import ObjectDetectionArray
from .arrays_classes.regression import RegressionArray
from .arrays_classes.scaler import ScalerArray
from .arrays_classes.segmentation import SegmentationArray
from .arrays_classes.speech_2_text import Speech2TextArray
from .arrays_classes.text_2_speech import Text2SpeechArray
from .arrays_classes.text_segmentation import TextSegmentationArray
from .arrays_classes.timeseries import TimeseriesArray
from .arrays_classes.tracker import TrackerArray
from .arrays_classes.raw import RawArray
from .preprocessing import CreatePreprocessing


class CreateArray(object):

    def __init__(self):
        self.image = ImageArray()
        self.audio = AudioArray()
        self.classification = ClassificationArray()
        self.text = TextArray()
        self.video = VideoArray()
        self.discriminator = DiscriminatorArray()
        self.generator = GeneratorArray()
        self.noise = NoiseArray()
        self.object_detection = ObjectDetectionArray()
        self.regression = RegressionArray()
        self.scaler = ScalerArray()
        self.segmentation = SegmentationArray()
        self.speech_2_text = Speech2TextArray()
        self.text_2_speech = Text2SpeechArray()
        self.text_segmentation = TextSegmentationArray()
        self.timeseries = TimeseriesArray()
        self.tracker = TrackerArray()
        self.raw = RawArray()

    def __getattr__(self, item: str):
        array_class = item.split('_')[-1]
        method = item.split('_')[0]
        return self.__dict__[array_class].__getattribute__(method)

    def execute_array(self, array_class: str, sources: Any, **options):
        if not isinstance(sources, list):
            sources = [sources]
        executor = self.__dict__[array_class]
        out_array = []

        source, parameters = self.get_result_items(result=executor.prepare(sources, dataset_folder=None, **options))
        for sour in source:
            array, parameters = self.get_result_items(result=executor.create(sour, **parameters))
            parameters['preprocess'] = options.get('preprocess')
            out_array.append(executor.preprocess(array, **parameters))

        out_array = np.array(out_array)

        return out_array

    def execute(self, array_class: str, dataset_path: str, sources: dict):

        out_array = {}
        temp_array = {}

        instructions, preprocessing = self.get_array_params(dataset_path=dataset_path)

        for put_id, cols_names in instructions.items():
            temp_array[put_id] = {}
            concat_list = []
            for col_name, data in cols_names.items():
                data['preprocess'] = preprocessing.preprocessing[put_id][col_name]
                for elem in sources[put_id][col_name]:
                    array = self.execute_array(array_class=array_class, sources=elem, **data)
                    concat_list.append(array)
            out_array[put_id] = np.concatenate(concat_list, axis=0)

        return out_array

    @staticmethod
    def get_array_params(dataset_path: str):
        instructions: dict = {}

        check_path = os.path.join(dataset_path, "dataset.json")
        if not os.path.exists(check_path):
            check_path = os.path.join(dataset_path, 'config.json')

        with open(check_path, 'r') as cfg:
            data = json.load(cfg)

        for put_id in data.get('inputs', {}).keys():
            instructions[put_id] = {}
            for instr_json in os.listdir(os.path.join(dataset_path, 'instructions', 'parameters')):
                idx, *name = os.path.splitext(instr_json)[0].split('_')
                name = '_'.join(name)
                if put_id == idx:
                    with open(os.path.join(dataset_path, 'instructions', 'parameters', instr_json), 'r') as instr:
                        instructions[put_id].update([(f'{idx}_{name}', json.load(instr))])
        preprocessing: CreatePreprocessing = CreatePreprocessing(dataset_path)
        preprocessing.load_preprocesses(data.get('columns', {}))

        return instructions, preprocessing

    @staticmethod
    def get_result_items(result: dict):
        source = result.get('instructions')
        parameters = result.get('parameters')
        return source, parameters
