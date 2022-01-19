from terra_ai.cascades.input_blocks import Input, BaseInput
from terra_ai.cascades.output_blocks import Output, BaseOutput
from terra_ai.cascades.services_blocks import Service
from terra_ai.cascades.function_blocks import Function


class BlockClasses:
    InputData = Input
    OutputData = Output
    Service = Service
    Function = Function

    @staticmethod
    def add_block(block_config: dict):
        block_type = block_config.get("parameters").get("main").get("type")
        group = BlockClasses().__getattribute__(block_config.get("group"))
        if group not in [BlockClasses.InputData, BlockClasses.OutputData]:
            block_object = group().get(type_=block_type, **block_config.get("parameters").get("main"))
        else:
            if block_type == "Video" and block_config.get("parameters").get("main").get("switch_on_frame"):
                block_type = "VideoFrameInput"
            block_object = group().get(type_=block_type)
        return block_object

    @staticmethod
    def get_bind(cascade_blocks: dict, bind: int):
        for block in cascade_blocks:
            if block.get("id") == bind:
                return block
        return None

    @staticmethod
    def get(cascade_config: dict):
        cascade_blocks = cascade_config.get("blocks", [])
        blocks_ = {}
        for block in cascade_blocks:
            id_ = block.get("id")
            if id_ not in blocks_.keys():
                blocks_[id_] = BlockClasses.add_block(BlockClasses.get_bind(cascade_blocks, id_))
                if issubclass(blocks_[id_].__class__, BaseOutput):
                    blocks_["output"] = id_
                if issubclass(blocks_[id_].__class__, BaseInput):
                    blocks_["input"] = id_
            block_inputs = block.get("bind").get("up")
            for bind in block_inputs:
                if bind not in blocks_.keys():
                    blocks_[bind] = BlockClasses.add_block(BlockClasses.get_bind(cascade_blocks, bind))
                bind_type = BlockClasses.get_bind(cascade_blocks, bind).get("parameters").get("main").get("type")
                blocks_.get(id_).inputs[bind_type] = blocks_.get(bind)

        return blocks_


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
                             'parameters': {'main': {'type': 'Video', 'width': 640, 'height': 480}}},
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
                                                     'classes': None,
                                                     'colors': None}}},
                            {'id': 6, 'name': 'Функция 6',
                             'group': 'Function', 'bind':{'up': [3], 'down': [4]},
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

    blocks = BlockClasses.get(cascade_config=cascade_2)
    print(blocks)
    input_ = blocks.get(blocks["input"])
    output_ = blocks.get(blocks["output"])
    input_.set_source(["F:\\Работа\\UII\\terra_gui\\Usage\\datasets\\loaded\\terra\\bus_video_tracker\\sources\\1_video\\Videos\\video_a_[0-480].avi"])
    output_.set_inputs(input_)

    print(output_.execute())



