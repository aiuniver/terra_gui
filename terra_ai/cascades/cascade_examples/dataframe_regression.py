from terra_ai.cascades.create import json2cascade


main_block = json2cascade(r"/home/evgeniy/terra_gui/terra_ai/cascades/demo_panel/cascades_json/"
                          r"dataframe_regression.json")


while True:
    input_path = input()
    main_block(input_path)
    print(main_block[-1].out)
