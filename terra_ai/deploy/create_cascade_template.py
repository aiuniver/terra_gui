import sys
import os
import json

from pydantic.color import Color


class CascadeConfigCreator:

    @staticmethod
    def create_config(model_path: str, out_path: str):
        dataset_path = os.path.join(model_path, "dataset", "config.json")
        with open(dataset_path) as cfg:
            dataset_config = json.load(cfg)

        tags = dataset_config['tags'][1]['alias']

        cascade_json_path = f"./demo_panel_templates/{tags}.json"
        with open(cascade_json_path) as cfg:
            config = json.load(cfg)

        config = getattr(sys.modules.get(__name__), f"make_{tags}")(config, dataset_config,
                                                                    os.path.split(model_path)[-1])
        with open(out_path, 'w') as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def make_classification(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        return config

    @staticmethod
    def make_segmentation(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        config['cascades']['2']['params']['num_class'] = dataset_config['outputs']['2']['num_classes']
        config['cascades']['2']['params']['classes_colors'] = [Color(i).as_rgb_tuple() for i in
                                                               dataset_config['outputs']['2']['classes_colors']]

        return config


if __name__ == "__main__":
    config = CascadeConfigCreator()
    config.create_config("C:\\Users\\Леккс\\AppData\\Local\\Temp\\tai-project\\training\\mnist",
                         "C:\\Users\\Леккс\\AppData\\Local\\Temp\\tai-project\\training")
