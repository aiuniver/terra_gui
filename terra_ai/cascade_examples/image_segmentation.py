from terra_ai.common import make_path
from terra_ai.cascades.create import json2cascade


path = make_path("test_example/cascades_example_segmentation_image.json")
main_block = json2cascade(path)


while True:
    input_path, output_path = input().split()
    main_block(input_path, output_path)
