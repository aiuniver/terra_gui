import os
import pathlib

from cascades.create import json2cascade


def predict(input_path):
    model = ""
    for path in os.listdir(str(pathlib.Path.cwd())):
        if path.endswith("cascade"):
            model = path
    main_block = json2cascade(model)
    main_block(input_path=input_path)
    return print(main_block.cascade_block[0][2].out)
