from terra_ai.cascades.create import json2cascade

import json
import os


path = r"C:\Users\Nill\Documents\terra_gui\terra_ai\cascades\demo_panel\cascades_json\timeseries.json"
main_block = json2cascade(path)

with open(path) as cfg:
    config = json.load(cfg)

dataset_path = os.path.join(config['cascades']['model']['model'], "dataset", "config.json")

with open(dataset_path) as cfg:
    config = json.load(cfg)

columns = [i[2:] for i in config['columns']['1'].keys()]

out_columns = [i[2:] for i in config['columns']['2'].keys()]

while True:
    input_path = input()
    data = next(main_block.input(input_path))[columns]

    main_block(input_path)
    out = {'source': {}, 'predict': {}}

    for col in columns:
        out['source'][col] = data[col].tolist()

    for i, col in enumerate(out_columns):
        out['predict'][col] = [out['source'][col] if col in out['source'].keys() else [],
                               main_block[-1].out[0][:, :, i].tolist()[0]]
    print(out)
