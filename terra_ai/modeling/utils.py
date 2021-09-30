import importlib
import json
import os

import graphviz  # conda install python-graphviz
import numpy as np
import tensorflow.keras.layers
import tensorflow_addons
from tensorflow import TensorShape

from terra_ai import customLayers
from terra_ai.data.modeling import layers

""" Официальная документация graphviz - https://www.graphviz.org/doc/info/shapes.html """

import yaml
from dataclasses import dataclass
# from terra_ai.trlayers import LayersDef
import copy

__version__ = 0.059


# for loading tuple type from file.yaml
class SafeLoader(yaml.SafeLoader):
    """
    Проблемные моменты: - не удается перенести в YAML объект Adam, нужно бы изменить форму записи на строковую 'Adam'
    во всех подаваемых классах - при подгрузке tuple из стороннего файла YAML выдается ошибка u'tag:yaml.org,
    2002:python/tuple'
    """

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", SafeLoader.construct_python_tuple
)


@dataclass
class ClassLoader:
    """
    Save and Load class from .yaml files
    Main using for saving and loading models
    """

    @staticmethod
    def dataclass_to_yaml(class_to_write, file_path):
        """Saving dataclass to yaml

        Arguments:
            class_to_write (object):    dataclass object
            file_path (str):            filename with path
        """
        attr_list = {}
        for attribute in dir(class_to_write):
            # if attribute doesn't have '__' in the beginning of name and not a function
            if (str(attribute)[:2] != '__') and ("<class 'function'>" != str(type(getattr(class_to_write, attribute)))):
                attr_list[attribute] = getattr(class_to_write, attribute)
        # write attribute dictionary to local .yaml file
        with open(f"{file_path}", 'w') as yaml_file:
            yaml.dump(dict(attr_list), yaml_file)
        pass

    @staticmethod
    def yaml_to_dataclass(file_path):
        """Loading dataclass from yaml
        Arguments:
            file_path (str):      filename with path
        """

        @dataclass
        class ClassToFill:
            """Create empty class for loading external data with varibles and attributes"""

            pass

        # read dictionary of attributes from .yaml file
        with open(f"{file_path}", "r") as file:
            template = yaml.load(file, Loader=SafeLoader)

        # for each attribute and its value from .yaml file
        for key, value in list(template.items()):
            setattr(ClassToFill, key, value)

        return ClassToFill

    # TODO: Проверить подгрузку параметров слоев кераса по дефолту
    # TODO: Добавить комментарии о проблемах

    def nn_plan_load(self, file_path):
        @dataclass
        class MyPlan():
            pass

        with open(f"{file_path}", "r") as file:
            template = yaml.load(file, Loader=SafeLoader)

        dummy = MyPlan()
        for key, value in list(template.items()):
            setattr(dummy, key, value)
        return dummy

    def nn_plan_save(self, my_plan, filepath) -> None:

        layers_def_keywords = [
            "framework",
            "input_datatype",
            "plan_name",
            "num_classes",
            "input_shape",
            "output_shape",
            "plan",
            "model_schema",
        ]
        if os.path.isfile(filepath):
            raise Exception("File with the same name already exists")

        # write exceptions
        layers_def_attrs = list(dir(LayersDef))
        layers_def_temp = copy.copy(layers_def_attrs)
        for keyword in layers_def_keywords:
            if keyword in layers_def_temp:
                layers_def_attrs.remove(keyword)

        attr_list = {}
        # and attribute not in dir(LayersDef)
        for attribute in dir(my_plan):
            # if attribute doesn't have '__' in the beginning of name and not a function
            if (
                    (str(attribute)[:2] != "__")
                    and ("<class 'function'>" != str(type(getattr(my_plan, attribute))))
                    and attribute not in layers_def_attrs
            ):
                attr_list[attribute] = getattr(my_plan, attribute)

        # write attribute dictionary to local .yaml file
        with open(f"{filepath}", "w") as yaml_file:
            yaml.dump(dict(attr_list), yaml_file)

        # check if saved correctly
        loaded_plan = self.nn_plan_load(filepath)
        # for i in dir(my_plan):
        #     if i not in dir(loaded_plan):
        #         print("Error: failed to save: ", i)
        #     if getattr(my_plan, i) != getattr(loaded_plan, i) and str(i)[:2] != "__":
        #         print("Error: failed to save: ", i)
        pass


class ModelLoader(ClassLoader):
    """
        Inherits from ClassLoader class
        Methods return information about YAML files and their contents
        from the directory specified in the variable named "file_path"
    """

    def __init__(self, file_path):
        self.file_path = file_path

    # @functools.lru_cache()
    def load_files(self):
        """
            Returns dictionary, where key is a YAML file name and value is dictionary of the file content
        """
        self.plans_dict = {}
        for i, el in enumerate(os.listdir(self.file_path)):
            with open(f"{self.file_path + el}", 'r') as file:
                self.plans_dict.update({el: yaml.load(file, Loader=yaml.Loader)})
        return self.plans_dict

    def get_files_names(self):
        """
            Returns list of the YAML files names
        """
        return list(self.plans_dict.keys())

    def get_input_types(self):
        """
            Returns list of the models input types
        """
        input_types = []
        for el in self.plans_dict:
            input_types.append(self.plans_dict[el]['input_datatype'])
        return input_types

    def get_input_shapes(self):
        """
            Returns list of the models input shapes
        """
        input_shapes = []
        for el in self.plans_dict:
            input_shapes.append(self.plans_dict[el]['input_shape'])
        return input_shapes

    def get_models_names(self):
        """
            Returns list of the models names
        """
        models_names = []
        for el in self.plans_dict:
            models_names.append(self.plans_dict[el]['plan_name'])
        return models_names

    def get_model_preview(self):
        """
            Returns dictionary, where key is the name of the YAML file and value is a list of main model info
            that the methods above return: input types, input shapes, models names
        """
        key_list = []
        model_preview = {}
        for i, el in enumerate(self.plans_dict):
            key_list.append([self.plans_dict[el]['input_datatype'], self.plans_dict[el]['input_shape'], \
                             self.plans_dict[el]['plan_name']])
            model_preview.update({el: key_list[i]})
        return model_preview


# noinspection PyUnresolvedReferences
class ModelVisualizer:
    """Create visualization of ModelPlan"""

    def __init__(self):
        # super(ModelVisualizer, self).__init__()
        self.save_path = "/"  # сделал по умолчанию, при необходимости можно присвоить свой путь
        pass

    # взято из кода plot_model, проверка на наличие установленного graphviz
    if graphviz is None:
        message = (
            "Failed to import graphviz. You must `pip install graphviz` "
            "and install graphviz (https://graphviz.org/download/) to OS, ",
            "for `pydotprint` to work.",
        )
        # print(message)

    def plot_nnmodel(self, adv_plan, verbose=1, show_size=0, file_path='') -> None:

        """
        Plot nnmodel from plan

        Args:
            model_plan (object):    obj, model from self.model_plan sequences
            verbose (int):          0, 1 or 2,  verbose=0 - show only layers names;
                                    verbose=1 - show layers names and input/output shapes;
                                    verbose=2 - show layers names, input/output shapes and layer parameters;
            show_size (int):        0, 1 or 2,  show_size=0 - show image with width=5;
                                    show_size=1 - show image with width=10;
                                    show_size=2 - show image with width=15;
                                    image lenth is calculated as lenth_default * width / width_default
            file_path (str):        if any file_path in string, method save file to that location with this name
            mode (str):             'standard' - without colors, except error marks
                                    'filled' - node filled with determined colors
                                    'clicked' - border of nodes with determined colors
        """

        # dot parameters definition
        dot = graphviz.Digraph(format="png")
        dot.attr("node", shape="record", penwidth="0")

        # for each layer name
        for i in range(len(adv_plan)):

            if adv_plan[i][8] != 'Pass':
                # print(f'Problem with layer {adv_plan[i][0]}: {adv_plan[i][8]}')
                layer_comm = f'<br/>{adv_plan[i][8]}'
            else:
                layer_comm = ''

            label_str = ''

            # show only labels
            if verbose == 0:
                label_str = f'<<table border="0" cellborder="0" cellspacing="0" cellpadding="4" BORDER = "1">' \
                            f'<tr><td width="200"><b>' \
                            f'<font point-size="20" {adv_plan[i][7]}>' \
                            f'{adv_plan[i][0]}{layer_comm}</font></b></td></tr></table>>'

            # show labels and shapes
            if verbose == 1:
                label_str = f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" BORDER = "1"><tr>' \
                            f'<td rowspan="2" width="200" ><b>' \
                            f'<font point-size="20" {adv_plan[i][7]}>{adv_plan[i][0]}{layer_comm}</font>' \
                            f'</b></td>' \
                            f'<td><font  point-size="14" {adv_plan[i][6]}>Input shape=' \
                            f'{adv_plan[i][4][0] if len(adv_plan[i][4]) == 1 else adv_plan[i][4]}' \
                            f'</font></td></tr><tr><td><font point-size="14" {adv_plan[i][7]}>' \
                            f'Output shape={adv_plan[i][5][0]}' \
                            f'</font></td></tr></table>>'

            # show labels, shapes and layer parameters
            if verbose == 2:
                if adv_plan[i][3] == {}:
                    label_str = f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4" BORDER = "1"><tr>' \
                                f'<td rowspan="2" width="200"><b>' \
                                f'<font point-size="20" {adv_plan[i][7]}>{adv_plan[i][0]}{layer_comm}</font>' \
                                f'</b></td>' \
                                f'<td><font point-size="14" {adv_plan[i][6]}>Input shape=' \
                                f'{adv_plan[i][4][0] if len(adv_plan[i][4]) == 1 else adv_plan[i][4]}' \
                                f'</font></td></tr>' \
                                f'<tr><td><font point-size="14" {adv_plan[i][7]}>' \
                                f'Output shape={adv_plan[i][5][0]}' \
                                f'</font></td></tr></table>>'
                else:
                    param_str = ''
                    for k, v in adv_plan[i][3].items():
                        param_str += f'{k} = {v}<br/>'
                    param_str = param_str[:-5]

                    label_str = f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4"><tr>' \
                                f'<td colspan="2" width="350"><b>' \
                                f'<font point-size="20" {adv_plan[i][7]}>{adv_plan[i][0]}{layer_comm}</font>' \
                                f'</b></td></tr><tr><td><font point-size="14" {adv_plan[i][6]}>Input shape=' \
                                f'{adv_plan[i][4][0] if len(adv_plan[i][4]) == 1 else adv_plan[i][4]}' \
                                f'</font></td><td rowspan="2">{param_str}</td></tr><tr>' \
                                f'<td><font point-size="14" {adv_plan[i][7]}>' \
                                f'Output shape={adv_plan[i][5][0]}</font></td></tr></table>>'

            dot.node(str(adv_plan[i][0]), label=label_str)

            # for each output of created node
            for out in adv_plan[i][2]:
                # through all layers inputs
                for j in range(len(adv_plan)):
                    # look for where each output of created node is an input of other node
                    if out in adv_plan[j][1]:
                        # and plot the edge
                        dot.edge(str(adv_plan[i][0]), str(adv_plan[j][0]))

        dot.graph_attr["rankdir"] = "TB"

        # save image file to directory
        if file_path != "":
            dot.render(file_path, cleanup=True, view=True)

        # if ask to show image in terminal
        else:
            from PIL import Image
            import matplotlib.pyplot as plt
            import os

            dot.render("model", cleanup=True, view=False)
            img = Image.open("model.png")

            if show_size == 0:
                width = 5
                plt.figure(figsize=((width, img.size[1] * width / img.size[0])))
            if show_size == 1:
                width = 10
                plt.figure(figsize=((width, img.size[1] * width / img.size[0])))
            if show_size == 2:
                width = 15
                plt.figure(figsize=((width, img.size[1] * width / img.size[0])))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            os.remove('model.png')
        pass


if __name__ == "__main__":
    pass
