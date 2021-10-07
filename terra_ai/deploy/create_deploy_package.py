import pathlib
import shutil
import sys
import os
import json

from pydantic.color import Color


class CascadeCreator:

    def create_config(self, model_path: str, out_path: str, func_name: str):
        out_path = os.path.join(out_path, "deploy")
        if func_name == "text_segmentation":
            dataset_path = os.path.join(model_path, "dataset", "instructions", "parameters", f"2_{func_name}.json")
        else:
            dataset_path = os.path.join(model_path, "dataset", "config.json")
        with open(dataset_path) as cfg:
            dataset_config = json.load(cfg)

        # tags = dataset_config['tags'][1]['alias']
        # if dataset_config["tags"][0]["alias"] == "text" and tags != "text_segmentation":
        #     tags = f"text_{tags}"
        # elif dataset_config["tags"][0]["alias"] != "text":
        #     tags = f"{dataset_config['tags'][0]['alias']}_{tags}"
        # if tags == "text_segmentation":
        #     dataset_path = os.path.join(model_path, "dataset", "instructions", "parameters", f"2_{tags}.json")
        #     with open(dataset_path) as cfg:
        #         dataset_config = json.load(cfg)

        cascade_json_path = f"terra_ai/deploy/demo_panel_templates/{func_name}.json"
        with open(cascade_json_path) as cfg:
            config = json.load(cfg)

        config = getattr(self, f"make_{func_name}")(config, dataset_config, os.path.split(model_path)[-1])
        with open(os.path.join(out_path, f"{os.path.split(model_path)[-1]}.cascade"), 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @staticmethod
    def make_image_classification(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        return config

    @staticmethod
    def make_text_classification(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        return config

    @staticmethod
    def make_audio_classification(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        return config

    @staticmethod
    def make_video_classification(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        return config

    @staticmethod
    def make_image_segmentation(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        config['cascades']['2']['params']['num_class'] = dataset_config['outputs']['2']['num_classes']
        config['cascades']['2']['params']['classes_colors'] = [Color(i).as_rgb_tuple() for i in
                                                               dataset_config['outputs']['2']['classes_colors']]

        return config

    @staticmethod
    def make_text_segmentation(config, dataset_config, model):
        config['cascades']['model']['model'] = model
        config['cascades']['2']['params']['open_tag'] = dataset_config['open_tags']
        config['cascades']['2']['params']['close_tag'] = dataset_config['close_tags']

        return config

    @staticmethod
    def copy_package(training_path):
        deploy_path = os.path.join(training_path, "deploy")
        if os.path.exists(os.path.join(deploy_path, "cascades")):
            shutil.rmtree(os.path.join(deploy_path, "cascades"), ignore_errors=True)
        if os.path.exists(os.path.join(deploy_path, "model")):
            shutil.rmtree(os.path.join(deploy_path, "model"), ignore_errors=True)
        shutil.copytree("terra_ai/cascades",
                        os.path.join(deploy_path, "cascades"),
                        ignore=shutil.ignore_patterns("demo_panel", "cascades"))
        shutil.copytree(os.path.join(training_path, "model"),
                        os.path.join(deploy_path, "model"),
                        ignore=shutil.ignore_patterns("deploy_presets", "interactive.history",
                                                      "config.presets", "config.train", "log.history"))
        shutil.copyfile("terra_ai/datasets/preprocessing.py",
                        os.path.join(deploy_path, "cascades", "preprocessing.py"))

    @staticmethod
    def copy_script(training_path, function_name):
        deploy_path = os.path.join(training_path, "deploy")
        shutil.copyfile(f"terra_ai/deploy/deploy_scripts/{function_name}.py",
                        os.path.join(deploy_path, "script.py"))


if __name__ == "__main__":
    config = CascadeCreator()
    config.create_config("C:\\Users\\Леккс\\AppData\\Local\\Temp\\tai-project\\training\\my_cars_new",
                         "C:\\Users\\Леккс\\AppData\\Local\\Temp\\tai-project\\training")
    config.copy_package("C:\\Users\\Леккс\\AppData\\Local\\Temp\\tai-project\\training")
