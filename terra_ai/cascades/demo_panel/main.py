import sys
import os
import json

from terra_ai.cascades.common import ROOT_PATH, make_path
from pydantic.color import Color


def make_classification(config, dataset_config, model):
    config['cascades']['model']['model'] = model
    return config


def make_segmentation(config, dataset_config, model):
    config['cascades']['model']['model'] = model
    config['cascades']['2']['params']['num_class'] = dataset_config['outputs']['2']['num_classes']
    config['cascades']['2']['params']['classes_colors'] = [Color(i).as_rgb_tuple() for i in
                                                           dataset_config['outputs']['2']['classes_colors']]

    return config


def make_text_segmentation(config, dataset_config, model):
    config['cascades']['model']['model'] = model
    config['cascades']['2']['params']['open_tag'] = dataset_config['columns']['1']['1_text']['open_tags']
    config['cascades']['2']['params']['close_tag'] = dataset_config['columns']['1']['1_text']['open_tags']

    return config


def create_config(model, out_path):
    path = make_path(model)
    dataset_path = os.path.join(path, "dataset", "config.json")
    with open(dataset_path) as cfg:
        dataset_config = json.load(cfg)

    for i in dataset_config['columns'].keys():
        for file in dataset_config['columns'][i].keys():
            with open(os.path.join(path, "dataset/instructions/parameters/" + file + ".json"), 'r') as f:
                f = json.load(f)
                dataset_config['columns'][i][file].update(f)

    tags = dataset_config['tags'][1]['alias']

    cascade_json_path = os.path.join(ROOT_PATH, 'terra_ai', 'cascades', "demo_panel", "cascades_json", tags + ".json")
    with open(cascade_json_path) as cfg:
        config = json.load(cfg)

    config = getattr(sys.modules.get(__name__), f"make_{tags}")(config, dataset_config, model)
    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    input_path, output_path = input().split()

    create_config(input_path, output_path)
