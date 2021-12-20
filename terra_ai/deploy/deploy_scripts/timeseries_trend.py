import json
import os


def predict(input_path):
    model = ""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for path in os.listdir(current_dir):
        if path.endswith("cascade"):
            model = os.path.join(current_dir, path)
    main_block = json2cascade(model)
    columns, out_columns = get_params(os.path.join(current_dir, model.split(".")[0]))

    data = next(main_block.input(input_path))[columns]
    main_block(input_path)
    out = {'source': {}, 'predict': {}}

    for col in columns:
        out['source'][col] = data[col].tolist()

    for i, col in enumerate(out_columns):
        out['predict'][col] = [out['source'][col] if col in out['source'].keys() else [],
                               main_block[-1].out[0][i]]

    return print(out)


def get_params(config_path):
    dataset_path = os.path.join(config_path, "dataset.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(config_path, "dataset", "config.json")
    with open(dataset_path) as cfg:
        config = json.load(cfg)

    columns = [i[2:] for i in config['columns']['1'].keys()]
    out_columns = [i[2:] for i in config['columns']['2'].keys()]

    return columns, out_columns


if __name__ in ["__main__", "script"]:
    from cascades.create import json2cascade
else:
    from .cascades.create import json2cascade