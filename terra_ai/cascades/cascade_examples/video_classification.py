from terra_ai.cascades.create import json2cascade


main_block = json2cascade(r"/home/evgeniy/terra_gui/terra_ai/cascades/demo_panel/cascades_json/video_classification"
                          r".json")

while True:
    input_path = input()
    # print(input_path)
    main_block(input_path)
    # print(main_block[-1].out)
    # print(main_block.cascade_block[0][2].out)  # [0] - модель классификация [2] - постпроцесс модели .out - выход
    # [[('Мерседес', 99), ('Рено', 0), ('Феррари', 0)]]
