from terra_ai.cascades.common import make_path
from terra_ai.cascades.create import json2cascade


main_block = json2cascade(r"/home/evgeniy/terra_gui/terra_ai/cascades/demo_panel/cascades_json/speech_to_text.json")

while True:
    input_path, output = input().split()
    main_block(input_path, output)
    print(main_block[1].out)

