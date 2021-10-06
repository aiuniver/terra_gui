from terra_ai.cascades.common import make_path
from terra_ai.cascades.create import json2cascade


path = make_path("C:\\PycharmProjects\\terra_gui\\test_example\\moloko.json")
main_block = json2cascade(path)

while True:
    input_path, output_path = input().split()
    main_block(input_path, output_path)
    # print(main_block.cascade_block[0][2].out)  # [0] - модель классификация [2] - постпроцесс модели .out - выход
    # [[('Мерседес', 99), ('Рено', 0), ('Феррари', 0)]]
