import datetime
import time

from .input_blocks import Input, BaseInput
from .model_blocks import Model, BaseModel
from .output_blocks import Output, BaseOutput
from .services_blocks import Service, DeepSort, BaseService
from .function_blocks import Function, PlotBboxes
from terra_ai.data.cascades.blocks.extra import ObjectDetectionFilterClassesList


class BlockClasses:
    InputData = Input
    OutputData = Output
    Service = Service
    Function = Function
    Model = Model

    @staticmethod
    def add_block(block_config: dict, model_path: str):
        block_type = block_config.get("parameters").get("main").get("type")
        if block_type is None:
            block_type = block_config.get('group')
        group = BlockClasses().__getattribute__(block_config.get("group"))
        if group in [BlockClasses.InputData, BlockClasses.OutputData]:
            if block_type == "Video" and block_config.get("parameters").get("main").get("switch_on_frame"):
                block_type = "VideoFrameInput"
        if group == BlockClasses.Model:
            block_object = group().get(type_=block_type, **block_config.get("parameters").get("main"))
            block_object.set_path(model_path=model_path, save_path='', weight_path='')
            block_object.get_outputs()
            if 'yolo' in block_object.model_architecture:
                block_type = 'yolo'
            del block_object
        block_object = group().get(type_=block_type, **block_config.get("parameters").get("main"))
        return block_object

    @staticmethod
    def get_bind(cascade_blocks: dict, bind: int):
        for block in cascade_blocks:
            if block.get("id") == bind:
                return block
        return None

    @staticmethod
    def get(cascade_config: dict, model_path=None, save_path=None, weight_path=None):
        cascade_blocks = cascade_config.get("blocks", [])
        blocks_ = {"output": []}

        for block in cascade_blocks:
            id_ = block.get("id")
            if id_ not in blocks_.keys():
                blocks_[id_] = BlockClasses.add_block(BlockClasses.get_bind(cascade_blocks, id_),
                                                      model_path=model_path)
                if issubclass(blocks_[id_].__class__, BaseOutput):
                    blocks_["output"].append(id_)
                if issubclass(blocks_[id_].__class__, BaseInput):
                    blocks_["input"] = id_
            block_inputs = block.get("bind").get("up")
            for bind in block_inputs:
                if bind not in blocks_.keys():
                    blocks_[bind] = BlockClasses.add_block(BlockClasses.get_bind(cascade_blocks, bind),
                                                           model_path=model_path)
                bind_type = BlockClasses.get_bind(cascade_blocks, bind).get("parameters").get("main").get("type")
                blocks_.get(id_).inputs[bind_type] = blocks_.get(bind)

        for idx, block in blocks_.items():
            if issubclass(block.__class__, (BaseModel, BaseService, PlotBboxes)):
                block.set_path(model_path=model_path, save_path=save_path, weight_path=weight_path)

        return blocks_


class Cascade:
    def __init__(self, model_path=None, save_path=None, weight_path=None, **config):
        self.blocks = BlockClasses.get(cascade_config=config, model_path=model_path,
                                       save_path=save_path, weight_path=weight_path)
        self.input_block = self.blocks.get(self.blocks["input"])
        self.output_block = {output: self.blocks.get(output) for output in self.blocks.get("output")}

    def get_outputs(self):
        return {id_: list(out.inputs.values())[0].get_outputs() for id_, out in self.output_block.items()}

    def __prepare_data(self, sources, output_file=None):
        self.input_block.set_source(sources)
        for id_, output in self.output_block.items():
            output.set_inputs(self.input_block)
            output.set_out(output_file)

    def execute(self, sources, output_file=None):
        result = {}
        self.__prepare_data(sources, output_file)
        for id_, output in self.output_block.items():
            result[id_] = output.execute()
        return result


if __name__ == "__main__":
    cascade_1 = {'alias': 'no_name', 'name': 'NoName', 'image': None,
                 'blocks': [{'id': 1, 'name': 'Input block',
                             'group': 'InputData', 'bind': {'up': [], 'down': [3]},
                             'position': [-83, -223],
                             'parameters': {'main': {'type': 'Audio', 'switch_on_frame': True}}},
                            {'id': 2, 'name': 'Output block',
                             'group': 'OutputData', 'bind': {'up': [3], 'down': []},
                             'position': [-84, 108],
                             'parameters': {'main': {'type': 'Text', 'width': 640, 'height': 480}}},
                            {'id': 3, 'name': 'Сервис 3',
                             'group': 'Service', 'bind': {'up': [1], 'down': [2]},
                             'position': [-90, -48],
                             'parameters': {'main': {'group': 'SpeechToText',
                                                     'type': 'TinkoffAPI',
                                                     'max_age': 4, 'min_hits': 4,
                                                     'distance_threshold': 0.4,
                                                     'metric': 'euclidean',
                                                     'version': 'Small',
                                                     'render_img': False,
                                                     'max_dist': 0.2,
                                                     'min_confidence': 0.3,
                                                     'nms_max_overlap': 1,
                                                     'max_iou_distance': 0.7,
                                                     'deep_max_age': 70,
                                                     'n_init': 3,
                                                     'nn_budget': 100,
                                                     'api_key': '8MVfedSXtjrIZinLXh4s/d+MJqV00RTUz/vzMoUvIsA=',
                                                     'secret_key': 'zTgzHQKfj/3luLO8VHEeXqKLacztaECIEXUAf1QGdoQ=',
                                                     'max_alternatives': 3,
                                                     'do_not_perform_vad': True,
                                                     'profanity_filter': True,
                                                     'expiration_time': 60000,
                                                     'endpoint': 'stt.tinkoff.ru:443',
                                                     'language': 'ru',
                                                     'model_path': None}}}
                            ]
                 }

    cascade_2 = {'alias': 'no_name', 'name': 'NoName', 'image': None,
                 'blocks': [{'id': 1, 'name': 'Input block',
                             'group': 'InputData', 'bind': {'up': [], 'down': [3, 4, 5]},
                             'position': [-178, -322],
                             'parameters': {'main': {'type': 'Video', 'switch_on_frame': True}}},
                            {'id': 2, 'name': 'Output block',
                             'group': 'OutputData', 'bind': {'up': [5], 'down': []},
                             'position': [17, 276],
                             'parameters': {'main': {'type': 'Video', 'width': 1280, 'height': 720}}},
                            {'id': 3, 'name': 'Сервис 3',
                             'group': 'Service', 'bind': {'up': [1], 'down': [6]},
                             'position': [-310, -150],
                             'parameters': {'main': {'group': 'ObjectDetection',
                                                     'type': 'YoloV5',
                                                     'max_age': 4,
                                                     'min_hits': 4,
                                                     'distance_threshold': 0.4,
                                                     'metric': 'euclidean',
                                                     'version': 'Small',
                                                     'render_img': False,
                                                     'max_dist': 0.2,
                                                     'min_confidence': 0.3,
                                                     'nms_max_overlap': 1,
                                                     'max_iou_distance': 0.7,
                                                     'deep_max_age': 70,
                                                     'n_init': 3,
                                                     'nn_budget': 100,
                                                     'api_key': None,
                                                     'secret_key': None,
                                                     'max_alternatives': 3,
                                                     'do_not_perform_vad': True,
                                                     'profanity_filter': True,
                                                     'expiration_time': 60000,
                                                     'endpoint': None,
                                                     'language': 'ru',
                                                     'model_path': None}}},
                            {'id': 4, 'name': 'Сервис 4',
                             'group': 'Service', 'bind': {'up': [6, 1], 'down': [5]},
                             'position': [61, 68],
                             'parameters': {'main': {'group': 'Tracking',
                                                     'type': 'DeepSort',
                                                     'max_age': 4,
                                                     'min_hits': 4,
                                                     'distance_threshold': 0.4,
                                                     'metric': 'euclidean',
                                                     'version': 'Small',
                                                     'render_img': False,
                                                     'max_dist': 0.2,
                                                     'min_confidence': 0.3,
                                                     'nms_max_overlap': 1.0,
                                                     'max_iou_distance': 0.7,
                                                     'deep_max_age': 70,
                                                     'n_init': 3,
                                                     'nn_budget': 100,
                                                     'api_key': None,
                                                     'secret_key': None,
                                                     'max_alternatives': 3,
                                                     'do_not_perform_vad': True,
                                                     'profanity_filter': True,
                                                     'expiration_time': 60000,
                                                     'endpoint': None,
                                                     'language': 'ru',
                                                     'model_path': "F:\\Работа\\UII\\terra_gui\\Usage\\modeling\\weights\\deepsort.t7"}}},
                            {'id': 5, 'name': 'Функция 5',
                             'group': 'Function', 'bind': {'up': [4, 1], 'down': [2]},
                             'position': [308, 177],
                             'parameters': {'main': {'group': 'ObjectDetection',
                                                     'type': 'PlotBBoxes',
                                                     'change_type': 'int',
                                                     'shape': None,
                                                     'min_scale': 0,
                                                     'max_scale': 1,
                                                     'alpha': 0.5,
                                                     'score_threshold': 0.3,
                                                     'iou_threshold': 0.45,
                                                     'method': 'nms',
                                                     'sigma': 0.3,
                                                     'line_thickness': 1,
                                                     'filter_classes': 'person',
                                                     'class_id': 0,
                                                     'classes_colors': None,
                                                     'open_tag': None,
                                                     'close_tag': None,
                                                     'classes': ObjectDetectionFilterClassesList,
                                                     'colors': None}}},
                            {'id': 6, 'name': 'Функция 6',
                             'group': 'Function', 'bind': {'up': [3], 'down': [4]},
                             'position': [-232, -22],
                             'parameters': {'main': {'group': 'ObjectDetection',
                                                     'type': 'FilterClasses',
                                                     'change_type': 'int',
                                                     'shape': None,
                                                     'min_scale': 0,
                                                     'max_scale': 1,
                                                     'alpha': 0.5,
                                                     'score_threshold': 0.3,
                                                     'iou_threshold': 0.45,
                                                     'method': 'nms',
                                                     'sigma': 0.3,
                                                     'line_thickness': 1,
                                                     'filter_classes': ['person',
                                                                        'bicycle',
                                                                        'car',
                                                                        'motorcycle',
                                                                        'airplane'],
                                                     'class_id': 0,
                                                     'classes_colors': None,
                                                     'open_tag': None,
                                                     'close_tag': None,
                                                     'classes': None,
                                                     'colors': None}}}]}
    cascade_3 = {'alias': 'no_name', 'name': 'NoName', 'image': None,
                 'blocks': [{'id': 1, 'name': 'Input block',
                             'group': 'InputData',
                             'bind': {'up': [], 'down': [5, 6, 8]}, 'position': [-86, -232],
                             'parameters': {'main': {'type': 'Video', 'switch_on_frame': True}}},
                            {'id': 2, 'name': 'Output block',
                             'group': 'OutputData', 'bind': {'up': [6], 'down': []}, 'position': [166, 334],
                             'parameters': {'main': {'type': 'Video', 'width': 640, 'height': 480}}},
                            {'id': 5, 'name': 'Сервис 5',
                             'group': 'Service',
                             'bind': {'up': [1, 7], 'down': [6]}, 'position': [2, 132],
                             'parameters': {
                                 'main': {'group': 'Tracking', 'type': 'DeepSort',
                                          'max_age': 4, 'min_hits': 4, 'distance_threshold': 0.4,
                                          'metric': 'euclidean', 'version': 'Small', 'render_img': False,
                                          'max_dist': 0.2, 'min_confidence': 0.3, 'nms_max_overlap': 1.0,
                                          'max_iou_distance': 0.7, 'deep_max_age': 70, 'n_init': 3,
                                          'nn_budget': 100, 'api_key': None, 'secret_key': None,
                                          'max_alternatives': 3, 'do_not_perform_vad': True,
                                          'profanity_filter': True, 'expiration_time': 60000,
                                          'endpoint': None, 'language': 'ru', 'model_path': None}}},
                            {'id': 6, 'name': 'Функция 6',
                             'group': 'Function',
                             'bind': {'up': [1, 5], 'down': [2]}, 'position': [142, 224],
                             'parameters': {
                                 'main': {'group': 'ObjectDetection', 'type': 'PlotBboxes', 'change_type': 'int',
                                          'shape': None, 'min_scale': 0, 'max_scale': 1, 'alpha': 0.5,
                                          'score_threshold': 0.3, 'iou_threshold': 0.45, 'method': 'nms',
                                          'sigma': 0.3, 'line_thickness': 1, 'filter_classes': 'person',
                                          'class_id': 0, 'classes_colors': None, 'open_tag': None,
                                          'close_tag': None, 'classes': ObjectDetectionFilterClassesList,
                                          'colors': None}}},
                            {'id': 7, 'name': 'Функция 7',
                             'group': 'Function', 'bind': {'up': [8], 'down': [5]}, 'position': [-160, 34],
                             'parameters': {
                                 'main': {'group': 'ObjectDetection', 'type': 'FilterClasses', 'change_type': 'int',
                                          'shape': None, 'min_scale': 0, 'max_scale': 1, 'alpha': 0.5,
                                          'score_threshold': 0.3, 'iou_threshold': 0.45, 'method': 'nms',
                                          'sigma': 0.3, 'line_thickness': 1, 'filter_classes': ['person'],
                                          'class_id': 0, 'classes_colors': None, 'open_tag': None, 'close_tag': None,
                                          'classes': None, 'colors': None}}},
                            {'id': 8, 'name': 'Сервис 8',
                             'group': 'Service', 'bind': {'up': [1], 'down': [7]}, 'position': [-210, -64],
                             'parameters': {
                                 'main': {'group': 'ObjectDetection', 'type': 'YoloV5', 'max_age': 4,
                                          'min_hits': 4, 'distance_threshold': 0.4, 'metric': 'euclidean',
                                          'version': 'Small', 'render_img': False, 'max_dist': 0.2,
                                          'min_confidence': 0.3, 'nms_max_overlap': 1, 'max_iou_distance': 0.7,
                                          'deep_max_age': 70, 'n_init': 3, 'nn_budget': 100, 'api_key': None,
                                          'secret_key': None, 'max_alternatives': 3, 'do_not_perform_vad': True,
                                          'profanity_filter': True, 'expiration_time': 60000, 'endpoint': None,
                                          'language': 'ru', 'model_path': None}}}]}

    cascade_4 = {'alias': 'no_name', 'name': 'NoName', 'image': None,
                 'blocks': [{'id': 1, 'name': 'Input block',
                             'group': 'InputData',
                             'bind': {'up': [], 'down': [5, 6, 7, 8]}, 'position': [-86, -232],
                             'parameters': {'main': {'type': 'Video', 'switch_on_frame': True}}},
                            {'id': 2, 'name': 'Output block',
                             'group': 'OutputData',
                             'bind': {'up': [6], 'down': []}, 'position': [166, 334],
                             'parameters': {'main': {'type': 'Video', 'width': 640, 'height': 480}}},
                            {'id': 5, 'name': 'Сервис 5',
                             'group': 'Service',
                             'bind': {'up': [1, 8], 'down': [6]}, 'position': [2, 132],
                             'parameters': {'main': {'group': 'Tracking', 'type': 'DeepSort', 'max_age': 4,
                                                     'min_hits': 4, 'distance_threshold': 0.4, 'metric': 'euclidean',
                                                     'version': 'Small', 'render_img': False, 'max_dist': 0.2,
                                                     'min_confidence': 0.3, 'nms_max_overlap': 1.0,
                                                     'max_iou_distance': 0.7, 'deep_max_age': 70, 'n_init': 3,
                                                     'nn_budget': 100, 'api_key': None, 'secret_key': None,
                                                     'max_alternatives': 3, 'do_not_perform_vad': True,
                                                     'profanity_filter': True, 'expiration_time': 60000,
                                                     'endpoint': None, 'language': 'ru',
                                                     'model_path': "F:\\Работа\\UII\\terra_gui\\Usage\\modeling\\weights\\deepsort.t7"}}},
                            {'id': 6, 'name': 'Функция 6',
                             'group': 'Function',
                             'bind': {'up': [1, 5], 'down': [2]}, 'position': [142, 224],
                             'parameters': {'main': {'group': 'ObjectDetection', 'type': 'PlotBboxes',
                                                     'change_type': 'int', 'shape': None, 'min_scale': 0,
                                                     'max_scale': 1, 'alpha': 0.5, 'score_threshold': 0.3,
                                                     'iou_threshold': 0.45, 'method': 'nms', 'sigma': 0.3,
                                                     'line_thickness': 1, 'filter_classes': 'person',
                                                     'class_id': 0, 'classes_colors': None, 'open_tag': None,
                                                     'close_tag': None, 'classes': ObjectDetectionFilterClassesList, 'colors': None}}},
                            {'id': 7, 'name': 'Модель 7',
                             'group': 'Model',
                             'bind': {'up': [1], 'down': [8]}, 'position': [-228, -64],
                             'parameters': {'main': {'path': 'шахматы_пре', 'postprocess': True}}},
                            {'id': 8, 'name': 'Функция 8',
                             'group': 'Function',
                             'bind': {'up': [1, 7], 'down': [5]}, 'position': [-98, 48],
                             'parameters': {'main': {'group': 'ObjectDetection', 'type': 'PostprocessBoxes',
                                                     'change_type': 'int', 'shape': None, 'min_scale': 0,
                                                     'max_scale': 1, 'alpha': 0.5, 'score_threshold': 0.3,
                                                     'iou_threshold': 0.45, 'method': 'nms', 'sigma': 0.3,
                                                     'line_thickness': 1, 'filter_classes': 'person', 'class_id': 0,
                                                     'classes_colors': None, 'open_tag': None, 'close_tag': None,
                                                     'classes': ObjectDetectionFilterClassesList, 'colors': None}}}]}

    training_path = 'F:\\content'
    cascade = Cascade(**cascade_4, model_path=training_path)
    # valid = cascade.validate()
    # res = cascade.execute([
    #     "F:\\Работа\\UII\\terra_gui\\Usage\\datasets\\loaded\\terra\\bus_video_tracker\\sources\\1_video\\Videos\\video_c_[915-1395].avi"],
    #     "F:\\test_odv_model.webm")

    res = cascade.execute([
        "F:\\2_[3000-3600].mp4"],
        "F:\\test_odv_model.webm")

    # res = cascade.execute([
    #     "F:\\Работа\\UII\\terra_gui\\Usage\\datasets\\loaded\\terra\\smarthome\\sources\\1_audio\\1_Кондиционер\\cond1 - ¬«»¿∩_[0.0-1.0].wav"],
    #     "F:\\test_stt.txt")

    print(res)
