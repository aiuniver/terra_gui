import os


def predict(input_path):
    model = ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in os.listdir(current_dir):
        if path.endswith("cascade"):
            model = os.path.join(current_dir, path)
    main_block = json2cascade(model)
    main_block(input_path)
    out = main_block[0].out
    while len(out) == 1:
        out = out[0]
    return print(out)


if __name__ in ["__main__", "script"]:
    from cascades.create import json2cascade
else:
    from .cascades.create import json2cascade
