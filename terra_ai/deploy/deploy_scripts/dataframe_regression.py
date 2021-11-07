import os


def predict(input_path):
    model = ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in os.listdir(current_dir):
        if path.endswith("cascade"):
            model = os.path.join(current_dir, path)
    main_block = json2cascade(model)
    main_block(input_path=input_path)
    return print([str(main_block[-1].out[0].tolist()[0])])


if __name__ in ["__main__", "script"]:
    from cascades.create import json2cascade
else:
    from .cascades.create import json2cascade
