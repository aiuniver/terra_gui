import os
import json

from pydantic.color import Color
import numpy as np

from terra_ai.cascades.common import make_path
from terra_ai.cascades.create import json2cascade


path = make_path("C:\\PycharmProjects\\terra_gui\\test_example\\airplane_new.json") #terra_ai/cascades/demo_panel/cascades_json/image_segmentation.json
main_block = json2cascade(path)

with open(path) as cfg:
    config = json.load(cfg)

dataset_path = make_path(os.path.join(config['cascades']['model']['model'], "dataset", "config.json"))
with open(dataset_path) as cfg:
    config = json.load(cfg)

names = config['outputs']['2']['classes_names']
colors = [Color(i).as_rgb_tuple() for i in config['outputs']['2']['classes_colors']]

while True:
    input_path, output_path = input().split()
    main_block(input_path, output_path)
    mask = main_block[0][1].out
    sum_list = [np.sum(mask[:, :, :, i]) for i in range(mask.shape[-1])]
    # print(str([(names[i], colors[i]) for i, count in enumerate(sum_list) if count > 0]))
