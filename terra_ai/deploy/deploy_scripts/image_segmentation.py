import json
import os
import pathlib
import numpy as np

from pydantic.color import Color
from cascades.create import json2cascade


def predict(input_path, output_path):
    model = ""
    for path in os.listdir(str(pathlib.Path.cwd())):
        if path.endswith("cascade"):
            model = path
    main_block = json2cascade(model)
    names, colors = get_params(os.path.join(pathlib.Path.cwd(), model.split(".")[0]))
    main_block(input_path=input_path, output_path=output_path)
    mask = main_block[0][1].out
    sum_list = [np.sum(mask[:, :, :, i]) for i in range(mask.shape[-1])]
    print(str([(names[i], colors[i]) for i, count in enumerate(sum_list) if count > 0]))


def get_params(config_path):
    dataset_path = os.path.join(config_path, "dataset", "config.json")
    with open(dataset_path) as cfg:
        config = json.load(cfg)

    names = config['outputs']['2']['classes_names']
    colors = [Color(i).as_rgb_tuple() for i in config['outputs']['2']['classes_colors']]
    return names, colors
