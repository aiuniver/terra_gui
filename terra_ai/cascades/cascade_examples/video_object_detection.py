from terra_ai.cascades.create import json2cascade


path = r"/home/evgeniy/terra_gui/terra_ai/cascades/demo_panel/cascades_json/video_object_detection.json"
main_block = json2cascade(path)

while True:
    input_path, output_path = input().split()
    main_block(input_path, output_path)
