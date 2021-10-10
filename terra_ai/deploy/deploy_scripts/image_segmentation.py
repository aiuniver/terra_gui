import json
import os
import numpy as np

from pydantic.color import Color


def predict(input_path, output_path):
    model = ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in os.listdir(current_dir):
        if path.endswith("cascade"):
            model = os.path.join(current_dir, path)
    main_block = json2cascade(model)
    names, colors = get_params(os.path.join(current_dir, model.split(".")[0]))
    main_block(input_path=input_path, output_path=output_path)
    mask = main_block[0][1].out
    sum_list = [np.sum(mask[:, :, :, i]) for i in range(mask.shape[-1])]
    return print(
        str([(names[i], colors[i]) for i, count in enumerate(sum_list) if count > 0])
    )


def get_params(config_path):
    dataset_path = os.path.join(config_path, "dataset", "config.json")
    with open(dataset_path) as cfg:
        config = json.load(cfg)

    names = config["outputs"]["2"]["classes_names"]
    colors = [Color(i).as_rgb_tuple() for i in config["outputs"]["2"]["classes_colors"]]
    return names, colors


if __name__ in ["__main__", "script"]:
    from cascades.create import json2cascade
else:
    from .cascades.create import json2cascade
