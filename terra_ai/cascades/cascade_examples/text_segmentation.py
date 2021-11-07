from terra_ai.cascades.common import make_path
from terra_ai.cascades.create import json2cascade

import json
import os
from random import randrange


path = make_path("terra_ai/cascades/demo_panel/cascades_json/text_segmentation.json")
# print(path)
main_block = json2cascade(path)

with open(path) as cfg:
    config = json.load(cfg)

dataset_path = make_path(os.path.join(config['cascades']['model']['model'], "dataset", "config.json"))

with open(dataset_path) as cfg:
    config = json.load(cfg)

parameters_path = make_path(
    config['cascades']['model']['model'],
    "dataset/instructions/parameters/2_text_segmentation.json"
)

with open(parameters_path) as f:
    open_tags = json.load(f)['open_tags']

classes = config['outputs']['2']['classes_names']
color = lambda: (randrange(1, 255) for _ in range(3))

format_out = [(tag, class_name, color) for tag, class_name in zip(open_tags, classes)]

input_path, output_path = input().split()

with open(output_path, 'w') as f:
    f.write(str(format_out))

main_block(input_path)
# print(main_block[-1].out)

while True:
    input_path = input()
    main_block(input_path)
    # print(main_block[-1].out)
