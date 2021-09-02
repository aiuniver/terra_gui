import copy
import gc
import importlib
import sys

import networkx as nx
import numpy as np
import tensorflow
from dataclasses import dataclass

from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model

from terra_ai.data.modeling import layers
from terra_ai.data.modeling.extra import LayerGroupChoice, LayerTypeChoice

from terra_ai.data.modeling.model import ModelDetailsData

__version__ = 0.052

from terra_ai.data.modeling.layers.extra import (
    ModuleTypeChoice,
    LayerValidationMethodChoice,
    SpaceToDepthDataFormatChoice,
)


@dataclass
class TerraModel:
    plan_name = ""
    input_shape = {}
    plan = []
    block_plans = {}
    pass


def get_links(model_plan):
    start_row = []
    end_row = []
    uplinks = {}
    downlinks = {}
    all_indexes = []
    for layer in model_plan:
        if layer[3] == [-1]:
            start_row.append(layer[0])
        if not layer[4]:
            end_row.append(layer[0])
        all_indexes.append(layer[0])
        downlinks[layer[0]] = layer[4]
        uplinks[layer[0]] = layer[3]
    return start_row, uplinks, downlinks, all_indexes, end_row


def get_idx_line(model_plan):
    start_row, uplinks, downlinks, idx2remove, _ = get_links(model_plan)
    distribution = []  # distribution plan, show rows with layers

    for i in start_row:
        if uplinks[i] != [-1]:
            start_row.pop(start_row.index(i))

    for i in start_row:
        idx2remove.pop(idx2remove.index(i))
    distribution.append(start_row)

    # get other rows
    count = 1
    while idx2remove:
        count += 1
        row_idxs = []
        for idx in distribution[-1]:
            for downlink in downlinks.get(idx):
                if downlink not in row_idxs:
                    row_idxs.append(downlink)

        for link in row_idxs:
            if (
                len(uplinks.get(link)) > 1
                and len(set(idx2remove) & set(uplinks.get(link))) != 0
            ):
                # print(set(idx2remove) & set(uplinks.get(link)))
                row_idxs.pop(row_idxs.index(link))

        distribution.append(row_idxs)
        for idx in row_idxs:
            idx2remove.pop(idx2remove.index(idx))
        if count > 100:
            idx2remove = None
    idx_line = []
    for row in distribution:
        idx_line.extend(row)
    return idx_line


def reorder_plan(model_plan):
    idx_line = get_idx_line(model_plan)
    order_plan = []
    for idx in idx_line:
        for layer in model_plan:
            if idx == layer[0]:
                order_plan.append(layer)
                break
    return order_plan


def get_edges(model_plan, full_connection=False):
    edges = []
    for layer in model_plan:
        for link in layer[3]:
            if full_connection:
                if link == -1:
                    edges.append((layer[0], layer[0]))
                else:
                    edges.append((layer[0], link))
            else:
                edges.append((layer[0], link))
    return edges


def reformat_input_shape(input_shape):
    if len(input_shape) == 1:
        if input_shape[0][0]:
            input_shape = list(input_shape[0])
            input_shape.insert(0, None)
            return [tuple(input_shape)]
        else:
            return input_shape
    else:
        new_input = []
        for inp in input_shape:
            if inp[0]:
                inp = list(inp)
                inp.insert(0, None)
                new_input.append(tuple(inp))
            else:
                new_input.append(inp)
        return new_input


def get_layer_info(layer_strict, block_name=None):
    params_dict = layer_strict.parameters.merged
    if (
        layer_strict.group == LayerGroupChoice.input
        or layer_strict.group == LayerGroupChoice.output
    ):
        params_dict["name"] = f"{layer_strict.id}"
    elif block_name:
        params_dict["name"] = f"{block_name}_{layer_strict.name}"
    else:
        params_dict["name"] = f"{layer_strict.type}_{layer_strict.id}"
    return (
        layer_strict.id,
        layer_strict.type.value,
        params_dict,
        [-1]
        if not layer_strict.bind.up
        else [-1 if x is None else x for x in layer_strict.bind.up],
        [x for x in layer_strict.bind.down],
    )


def tensor_shape_to_tuple(tensor_shape):
    tuple_shape = []
    for dim in tensor_shape:
        tuple_shape.append(dim)
    return tuple(tuple_shape)


class ModelValidator:
    """Make validation of model plan"""

    def __init__(self, model, output_shape=None, **kwargs):
        self.validator = LayerValidation()
        self.model_plan = None
        self.filled_model = None
        self.maxwordcount = None

        self.all_indexes = []
        self.start_row = []
        self.end_row = []
        self.uplinks = {}
        self.downlinks = {}
        self.layers_config = {}
        self.layers_def = {}
        self.layers_state = {}
        self.layer_input_shapes = {}
        self.layer_output_shapes = {}
        self.model = model
        self.filled_model = model
        self.model_plan = TerraModel()
        self.model_plan.plan = []
        self.model_plan.plan_name = ""
        self.model_plan.input_shape = {}
        self.model_plan.block_plans = {}

        self.keras_code = ""
        self.valid = True

        self.output_shape = output_shape
        self.maxwordcount = kwargs.get("maxwordcount")

        self.val_dictionary = {}
        for layer in self.model.layers:
            self.val_dictionary[layer.id] = None
            self.layer_input_shapes[layer.id] = []
            self.layer_output_shapes[layer.id] = []
            self.layers_state[layer.id] = ""
            if layer.reference:
                for block in self.model.references:
                    if layer.reference == block.name:
                        block_def = {}
                        block_config = {}
                        for block_layer in block.details.layers:
                            block_def[
                                block_layer.id
                            ] = block_layer.parameters.defaults.merged
                            block_config[block_layer.id] = block_layer.parameters.config
                        self.layers_def[layer.id] = block_def
                        self.layers_config[layer.id] = block_config
                    break
            else:
                self.layers_def[layer.id] = layer.parameters.defaults.merged
                self.layers_config[layer.id] = layer.parameters.config
        self._build_model_plan()

    def _build_model_plan(self):
        # оставить описание
        self.model_plan.plan_name = self.model.alias
        for layer in self.model.layers:
            if layer.group == LayerGroupChoice.input:
                self.model_plan.input_shape[layer.id] = layer.shape.input
                self.layer_input_shapes[layer.id].extend(
                    reformat_input_shape(layer.shape.input)
                )
            self.layers_state[layer.id] = layer.group.value
            self.model_plan.plan.append(get_layer_info(layer))
            if layer.reference:
                for block in self.model.references:
                    if layer.reference == block.name:
                        block_plan = []
                        for block_layer in block.details.layers:
                            block_plan.append(
                                get_layer_info(block_layer, block_name=layer.name)
                            )
                        self.model_plan.block_plans[layer.id] = reorder_plan(block_plan)
                        break
        self._get_model_links()
        self._get_reorder_model()

    def _get_cycles_check(self):
        """
        Check if there are cycles in the structure

        Returns:
            val_dictionary, dict:   with or without comments about cycles
            valid, bool:            True if no cycles, otherwise False
        """

        edges = get_edges(self.model_plan.plan)

        di_graph = nx.DiGraph(edges)
        for cycle in nx.simple_cycles(di_graph):
            if cycle:
                self.valid = False
                comment = f"Layers {cycle} make a cycle! Please correct the structure!"
                for cycle_layer in cycle:
                    self.val_dictionary[cycle_layer] = comment

    def _get_full_connection_check(self):
        """
        Check if there separated layers or groups of layers in plan

        Returns:
            val_dictionary, dict:   with or without comments about separation
            valid, bool:            True if no separation, otherwise False
        """

        edges = get_edges(self.model_plan.plan, full_connection=True)
        di_graph = nx.DiGraph(edges)
        subgraphs = sorted(
            list(nx.weakly_connected_components(di_graph)),
            key=lambda subgraph: -len(subgraph),
        )

        if len(subgraphs) > 1:
            self.valid = False
            for group in subgraphs[1:]:
                for layer in group:
                    self.val_dictionary[
                        layer
                    ] = "Connection Error: layer is not connected to main part!"

    def _get_model_links(self):
        (
            self.start_row,
            self.uplinks,
            self.downlinks,
            self.all_indexes,
            self.end_row,
        ) = get_links(self.model_plan.plan)

    def _get_reorder_model(self):
        self.model_plan.plan = reorder_plan(self.model_plan.plan)

    def _get_input_shape_check(self):
        """Check empty input shapes"""
        input_layers = {}
        for layer in self.model_plan.plan:
            if layer[0] in self.model_plan.input_shape.keys():
                input_layers[layer[0]] = layer[2].get("name")

        for idx in self.start_row:
            if idx not in input_layers.keys():
                self.valid = False
                self.layer_input_shapes[idx].append(None)
                self.val_dictionary[
                    idx
                ] = "Input shape Error: layer does not have input shape!"

        # check if plan input shapes is not None
        for _id, shape in self.model_plan.input_shape.items():
            if not shape or None in shape:
                self.valid = False
                self.val_dictionary[_id] = "Input shape Error: layer does not have input shape!"

    def _get_output_shape_check(self):
        """Check compatibility of dataset's and results model output shapes"""
        if self.output_shape:
            outputs = []
            for layer in self.model_plan.plan:
                if layer[0] in self.output_shape.keys():
                    outputs.append(layer[0])
                    if (
                        self.output_shape[layer[0]][0]
                        != self.layer_output_shapes[layer[0]][0][1:]
                    ):
                        self.valid = False
                        self.val_dictionary[layer[0]] = (
                            f"Output shape Error: Expected output shape "
                            f"{self.output_shape[layer[0]][0]} "
                            f"but got output shape {self.layer_output_shapes[layer[0]][0][1:]}!"
                        )

            # check unspecified output layers
            for idx in self.end_row:
                if idx not in outputs:
                    self.valid = False
                    self.val_dictionary[
                        idx
                    ] = "Output shape Error: Unspecified output layer!"

    def _model_validation(self):
        """Full model modeling"""
        # check for cycles
        self._get_cycles_check()
        if not self.valid:
            return self.val_dictionary

        # check for full connection
        self._get_full_connection_check()
        if not self.valid:
            return self.val_dictionary

        # check for input shapes compatibility
        self._get_model_links()
        self._get_input_shape_check()
        if not self.valid:
            return self.val_dictionary

        # check layers
        for layer in self.model_plan.plan:
            if layer[1] == LayerTypeChoice.CustomBlock:
                output_shape, comment = self._custom_block_validation(
                    self.model_plan.block_plans[layer[0]],
                    self.layer_input_shapes.get(layer[0]),
                    self.layers_def[layer[0]],
                    self.layers_config[layer[0]],
                )
                if comment:
                    comment = f"Errors in block {layer[2].get('name', layer[0])}: {comment[:-2]}"
            else:
                output_shape, comment = self._layer_validation(
                    layer,
                    self.layer_input_shapes.get(layer[0]),
                    self.layers_def[layer[0]],
                    self.layers_config[layer[0]],
                )
            self.layer_output_shapes[layer[0]] = output_shape
            if comment:
                self.valid = False
                self.val_dictionary[layer[0]] = comment
            for downlink in self.downlinks[layer[0]]:
                self.layer_input_shapes[downlink].extend(output_shape)
        if not self.valid:
            return self.val_dictionary

        # check output shapes compatibility
        self._get_output_shape_check()
        return self.val_dictionary

    def _layer_validation(self, layer, layer_input_shape, defaults, config):
        self.validator.set_state(
            layer[1], layer_input_shape, layer[2], defaults, config
        )
        return self.validator.get_validated()

    def _custom_block_validation(self, block_plan, block_input_shape, defaults, config):
        """block modeling"""
        _, _, downlinks, _, end_row = get_links(block_plan)
        block_val_dict = {}
        block_input_shapes = {}
        block_output_shapes = {}
        for layer in block_plan:
            block_val_dict[layer[0]] = None
            block_input_shapes[layer[0]] = []
            if -1 in block_plan[layer[0]][3]:
                block_input_shapes[layer[0]].extend(block_input_shape)
            block_output_shapes[layer[0]] = []

        # check layers
        for layer in block_plan:
            output_shape, comment = self._layer_validation(
                layer,
                block_input_shapes[layer[0]],
                defaults[layer[0]],
                config[layer[0]],
            )
            block_output_shapes[layer[0]] = output_shape
            if comment:
                block_val_dict[layer[0]] = comment
            for downlink in downlinks[layer[0]]:
                block_input_shapes[downlink].extend(output_shape)

        block_output = []
        for idx in end_row:
            block_output.extend(block_output_shapes.get(idx))

        block_comment = ""
        for idx, value in block_val_dict.items():
            if value is not None:
                if value != "Input shape Error: received empty input shape!":
                    block_comment += (
                        f"block layer {block_plan[idx][2].get('name', idx)} - {value}, "
                    )

        return block_output, block_comment

    def compile_keras_code(self):
        """Create keras code from model plan"""

        self.keras_code = ""
        layers_import = {}
        name_dict = {}
        input_list = []
        output_list = []
        for layer in self.model_plan.plan:
            # керас код под block_plan пока не готов
            # layer_type = layer[1] if layer[1] != 'space_to_depth' else 'SpaceToDepth'

            if (
                layer[1] not in layers_import.values()
                and self.layers_config.get(layer[0]).module_type.value
                != ModuleTypeChoice.block_plan
            ):
                layers_import[layer[0]] = layer[1]

            if (
                self.layers_config.get(layer[0]).module_type.value
                == ModuleTypeChoice.block_plan
            ):
                for block_layer in self.model_plan.block_plans.get(layer[0]):
                    if block_layer[1] not in layers_import.values():
                        layers_import[block_layer[0]] = block_layer[1]

            if layer[0] in self.start_row:
                name_dict[layer[0]] = f"input_{layer[2].get('name')}"
                input_list.append(f"input_{layer[2].get('name')}")
            elif layer[0] in self.end_row:
                name_dict[layer[0]] = f"output_{layer[2].get('name')}"
                output_list.append(f"output_{layer[2].get('name')}")
            else:
                name_dict[layer[0]] = f"x_{layer[0]}"

        layers_str = ""
        for _id, layer_name in layers_import.items():
            # layer_type = i if i != 'space_to_depth' else 'SpaceToDepth'
            layers_str += (
                f"from {self.layers_config.get(_id).module.value} import {layer_name}\n"
            )
        layers_str = f"{layers_str}from tensorflow.keras.models import Model\n\n"

        inputs_str = ""
        for i in input_list:
            inputs_str += f"{i}, "
        inputs_str = f"[{inputs_str[:-2]}]"

        outputs_str = ""
        for i in output_list:
            outputs_str += f"{i}, "
        outputs_str = f"[{outputs_str[:-2]}]"

        def get_layer_str(layer, identificator="", block_uplinks=None):
            layer_str = ""
            if block_uplinks:
                block_uplinks[layer[0]] = f"{identificator}_{layer[1]}_{layer[0]}"

            if layer[1] == LayerTypeChoice.Input:
                layer_str = (
                    f"{block_uplinks[layer[0]] if block_uplinks else name_dict[layer[0]]} = "
                    f"{layer[1]}(shape={self.model_plan.input_shape[layer[0]]}, "
                    f"name='{layer[2].get('name')}')\n"
                )
            else:
                params = ""
                for key in layer[2].keys():
                    if key not in ["trainable", "output_layer"]:
                        if isinstance(layer[2][key], str):
                            params += f"{key}='{layer[2][key]}', "
                        else:
                            params += f"{key}={layer[2][key]}, "
                if len(layer[3]) == 1:
                    if block_uplinks:
                        uplink = f"{block_uplinks[layer[3][0]]}"
                    else:
                        uplink = f"{name_dict[layer[3][0]]}"
                else:
                    uplink = "["
                    for up in layer[3]:
                        if block_uplinks:
                            uplink += f"{block_uplinks[up]}, "
                        else:
                            uplink += f"{name_dict[up]}, "
                    uplink = f"{uplink[:-2]}]"

                if (
                    self.layers_config.get(layer[0]).module_type.value
                    == ModuleTypeChoice.tensorflow
                ):
                    layer_str = (
                        f"{block_uplinks[layer[0]] if block_uplinks else name_dict[layer[0]]} = "
                        f"{layer[1]}({uplink}, {params[:-2]})\n"
                    )
                elif (
                    self.layers_config.get(layer[0]).module_type.value
                    != ModuleTypeChoice.keras_pretrained_model
                ):
                    layer_str = (
                        f"{block_uplinks[layer[0]] if block_uplinks else name_dict[layer[0]]} = "
                        f"{layer[1]}({params[:-2]})({uplink})\n"
                    )
                elif (
                    self.layers_config.get(layer[0]).module_type.value
                    == ModuleTypeChoice.keras_pretrained_model
                ):
                    if "trainable" in layer[2].keys():
                        block_name = f"{block_uplinks[layer[0]] if block_uplinks else name_dict[layer[0]]}"
                        if layer[2].get("output_layer") == "last":
                            out_layer_str = f"{block_name}.output"
                        else:
                            out_layer_str = f"{block_name}.get_layer('{layer[2].get('output_layer')}.output'"
                        layer_str = (
                            f"\n{block_name} = {layer[1]}({params[:-2]})\n"
                            f"for layer in {block_name}.layers:\n"
                            f"    layer.trainable = {layer[3].get('trainable', False)}\n"
                            f"{block_name} = Model({block_name}.input, {out_layer_str}).output, "
                            f"name='{block_name}')\n"
                            f"{name_dict[layer[0]]} = {block_name}({uplink})\n\n"
                        )
                else:
                    pass
            return layer_str

        for layer in self.model_plan.plan:
            if layer[1] == LayerTypeChoice.CustomBlock:
                layer_str = ""
                block_uplinks = {-1: name_dict[layer[3][0]]}
                for block_layer in self.model_plan.block_plans.get(layer[0]):
                    layer_str += get_layer_str(
                        block_layer,
                        identificator=layer[2].get("name", ""),
                        block_uplinks=block_uplinks,
                    )
                layer_str = f"\n{layer_str}\n"
            else:

                layer_str = get_layer_str(layer)
            self.keras_code += layer_str

        if self.keras_code:
            self.keras_code = f"{layers_str}{self.keras_code[:-2]})"
            self.keras_code = (
                f"{self.keras_code}\n\nmodel = Model({inputs_str}, {outputs_str})"
            )

    def get_validated(self) -> (ModelDetailsData, dict):
        """Returns all necessary info about modeling"""

        self._model_validation()
        if self.valid:
            self.compile_keras_code()
        else:
            self.keras_code = None

        for idx, layer in enumerate(self.filled_model.layers):
            # fill inputs
            if layer.group == LayerGroupChoice.input:
                pass
            elif not self.layer_input_shapes.get(layer.id):
                self.filled_model.layers[idx].shape.input = []
            elif len(self.layer_input_shapes.get(layer.id)) == 1:
                self.filled_model.layers[idx].shape.input = [
                    self.layer_input_shapes.get(layer.id)[0][1:]
                    if self.layer_input_shapes.get(layer.id)[0]
                    else self.layer_input_shapes.get(layer.id)
                ]
            else:
                front_shape = []
                for shape in self.layer_input_shapes.get(layer.id):
                    if shape:
                        front_shape.append(shape[1:])
                    else:
                        front_shape.append(shape)
                self.filled_model.layers[idx].shape.input = front_shape

            # fill outputs
            if not self.layer_output_shapes.get(layer.id):
                self.filled_model.layers[idx].shape.output = []
            else:
                self.filled_model.layers[idx].shape.output = [
                    self.layer_output_shapes.get(layer.id)[0][1:]
                    if self.layer_output_shapes.get(layer.id)[0]
                    else self.layer_output_shapes.get(layer.id)
                ]

        self.filled_model.keras = self.keras_code
        return self.filled_model, self.val_dictionary

    def get_keras_model(self):
        mc = ModelCreator(self.model_plan, self.layers_config)
        return mc.create_model()


class LayerValidation:
    """Validate input shape, number uplinks and parameters compatibility"""

    def __init__(self):
        self.inp_shape: list = [None]
        self.layer_type: str = ""
        self.def_parameters: dict = {}
        self.layer_parameters: dict = {}
        self.num_uplinks: tuple = None
        self.input_dimension: tuple = None
        self.module: str = ""
        self.module_type: str = ""
        self.kwargs: dict = {}

    def set_state(self, layer_type, shape, parameters, defaults, config, **kwargs):
        """Set input data and fill attributes"""
        # print("LayerValidation set_state", layer_type, shape, parameters)
        self.layer_type = layer_type
        self.inp_shape = shape
        self.def_parameters = defaults
        self.layer_parameters = parameters
        self.kwargs = kwargs

        self.num_uplinks = (
            config.num_uplinks.value,
            config.num_uplinks.validation.value,
        )
        self.input_dimension = (
            config.input_dimension.value,
            config.input_dimension.validation.value,
        )
        self.module = importlib.import_module(config.module.value)
        self.module_type = config.module_type.value

    def get_validated(self):
        """Validate given layer parameters and return output shape and possible error comment"""
        error = self.primary_layer_validation()
        if error:
            return [None], error
        else:
            output_shape = [None]
            if (
                self.module_type == ModuleTypeChoice.keras
                or self.module_type == ModuleTypeChoice.terra_layer
                or self.module_type == ModuleTypeChoice.keras_pretrained_model
            ):
                try:
                    if self.layer_type == LayerTypeChoice.Input:
                        return self.inp_shape, None
                    if self.module_type == ModuleTypeChoice.keras_pretrained_model:
                        self.layer_parameters.pop("trainable")
                        if self.layer_parameters.get("name"):
                            self.layer_parameters.pop("name")
                    output_shape = [
                        tuple(
                            getattr(self.module, self.layer_type)(
                                **self.layer_parameters
                            ).compute_output_shape(
                                self.inp_shape[0]
                                if len(self.inp_shape) == 1
                                else self.inp_shape
                            )
                        )
                    ]
                    # LSTM and GRU can returns list of one tuple of tensorshapes
                    # code below reformat it to list of shapes
                    if (
                        len(output_shape) == 1
                        and type(output_shape[0][0]).__name__ == "TensorShape"
                    ):
                        new = []
                        for shape in output_shape[0]:
                            new.append(tensor_shape_to_tuple(shape))
                        return new, None

                    return output_shape, None
                except Exception:
                    return output_shape, self.parameters_validation()
            if self.module_type == ModuleTypeChoice.tensorflow:
                try:
                    inp_shape = (
                        self.inp_shape[0][1:]
                        if len(self.inp_shape) == 1
                        else [self.inp_shape[i][1:] for i in range(len(self.inp_shape))]
                    )
                    output = getattr(tensorflow.nn, self.layer_type)(
                        tensorflow.keras.layers.Input(shape=inp_shape),
                        **self.layer_parameters,
                    )
                    # print(type(tensor_shape_to_tuple(output.shape)))
                    return [tensor_shape_to_tuple(output.shape)], None
                except Exception:
                    return output_shape, self.parameters_validation()

    def get_problem_parameter(
        self, base_dict: dict, check_dict: dict, problem_dict, inp_shape, revert=False
    ):
        """check each not default parameter from check_dict by setting it in base_dict
        revert means set default parameter in layer parameters and need additional check if pass
        on initial layer parameters"""
        for param in base_dict.keys():
            val_dict = copy.deepcopy(base_dict)
            if val_dict.get(param) != check_dict.get(param):
                val_dict[param] = check_dict.get(param)
                try:
                    if (
                        self.module_type == ModuleTypeChoice.keras
                        or self.module_type == ModuleTypeChoice.terra_layer
                    ):
                        del val_dict["name"]
                        getattr(self.module, self.layer_type)(
                            **val_dict
                        ).compute_output_shape(
                            inp_shape[0] if len(inp_shape) == 1 else inp_shape
                        )

                    elif self.module_type == ModuleTypeChoice.tensorflow:

                        inp_shape = (
                            self.inp_shape[0][1:]
                            if len(self.inp_shape) == 1
                            else [
                                self.inp_shape[i][1:]
                                for i in range(len(self.inp_shape))
                            ]
                        )
                        getattr(tensorflow.nn, self.layer_type)(
                            tensorflow.keras.layers.Input(shape=inp_shape), **val_dict
                        )
                        # print(self.module, self.layer_type)
                    if revert:
                        try:
                            if (
                                self.module_type == ModuleTypeChoice.keras
                                or self.module_type == ModuleTypeChoice.terra_layer
                            ):
                                getattr(self.module, self.layer_type)(
                                    **base_dict
                                ).compute_output_shape(
                                    inp_shape[0] if len(inp_shape) == 1 else inp_shape
                                )
                            if self.module_type == ModuleTypeChoice.tensorflow:
                                getattr(self.module, self.layer_type)(
                                    tensorflow.keras.layers.Input(shape=inp_shape),
                                    **base_dict,
                                )
                        except ValueError as error:
                            problem_dict[param] = (base_dict.get(param), str(error))
                        except TypeError as error:
                            problem_dict[param] = (base_dict.get(param), str(error))
                except ValueError as error:
                    if not revert:
                        problem_dict[param] = (val_dict.get(param), str(error))
                except TypeError as error:
                    if not revert:
                        problem_dict[param] = (val_dict.get(param), str(error))
        return problem_dict

    def parameters_validation(self):
        """Parameter control comparing with default"""
        if isinstance(self.def_parameters, str):
            return self.def_parameters

        problem_params = {}
        problem_params = self.get_problem_parameter(
            base_dict=self.def_parameters,
            check_dict=self.layer_parameters,
            problem_dict=problem_params,
            inp_shape=self.inp_shape,
            revert=False,
        )
        problem_params = self.get_problem_parameter(
            base_dict=self.layer_parameters,
            check_dict=self.def_parameters,
            problem_dict=problem_params,
            inp_shape=self.inp_shape,
            revert=True,
        )
        comment = "Parameters Error: check the following parameters: "
        if problem_params:
            for key in problem_params.keys():
                if isinstance(problem_params[key][0], str):
                    comment += (
                        f"{key}='{problem_params[key][0]}' ({problem_params[key][1]}); "
                    )
                else:
                    comment += (
                        f"{key}={problem_params[key][0]} ({problem_params[key][1]}); "
                    )
            return comment[:-2]

    def primary_layer_validation(self):
        """Whole modeling for specific parameters, uplink number and input dimension"""
        comment = self.position_validation()
        if comment:
            return comment
        comment = self.input_dimension_validation()
        if comment:
            return comment
        comment = self.specific_parameters_validation()
        if comment:
            return comment
        else:
            return None

    def position_validation(self):
        """Validate number of uplinks"""
        if None in self.inp_shape:
            return "Input shape Error: received empty input shape!"
        elif (
            isinstance(self.num_uplinks[0], int)
            and self.num_uplinks[1] == LayerValidationMethodChoice.fixed
            and len(self.inp_shape) != self.num_uplinks[0]
        ):
            return (
                f"Position Error: Expected {self.num_uplinks[0]} "
                f"input shape{'s' if self.num_uplinks[0] > 1 else ''} but got {len(self.inp_shape)}!"
            )
        elif (
            isinstance(self.num_uplinks[0], int)
            and self.num_uplinks[1] == LayerValidationMethodChoice.minimal
            and len(self.inp_shape) < self.num_uplinks[0]
        ):
            return (
                f"Position Error: Expected {self.num_uplinks[0]} or greater "
                f"input shape{'s' if self.num_uplinks[0] > 1 else ''} but got {len(self.inp_shape)}!"
            )
        elif (
            isinstance(self.num_uplinks[0], tuple)
            and self.num_uplinks[1]
            not in [
                LayerValidationMethodChoice.dependence_tuple2,
                LayerValidationMethodChoice.dependence_tuple3,
            ]
            and len(self.inp_shape) not in self.num_uplinks[0]
        ):
            return (
                f"Position Error: Expected one of {self.num_uplinks} "
                f"input shapes but got {len(self.inp_shape)}!"
            )
        else:
            return None

    def input_dimension_validation(self):
        """Dimention compatibility of first_shape shape"""
        if len(self.inp_shape) > 1:
            for shape in self.inp_shape[1:]:
                if len(self.inp_shape[0]) != len(shape):
                    return f"Input shape Error: Input shapes have different sizes {self.inp_shape}!"
            axis = self.layer_parameters.get("axis", None)
            if axis:
                first_shape = list(self.inp_shape[0])
                first_shape.pop(axis)
                for shape in self.inp_shape[1:]:
                    shape = list(shape)
                    shape.pop(axis)
                    if shape != first_shape:
                        return (
                            f"Input shape Error: required inputs with matching shapes except "
                            f"for the concat axis {axis} but received {self.inp_shape}!"
                        )
            else:
                for shape in self.inp_shape[1:]:
                    if shape != self.inp_shape[0]:
                        return (
                            f"Input shape Error: All input shapes must be "
                            f"the same but received {self.inp_shape}!"
                        )
        else:
            if (
                isinstance(self.input_dimension[0], int)
                and self.input_dimension[1] == LayerValidationMethodChoice.fixed
                and len(self.inp_shape[0]) != self.input_dimension[0]
            ):
                return (
                    f"Input dimension Error: Expected dim = {self.input_dimension[0]} "
                    f"but got dim={len(self.inp_shape[0])}!"
                )
            elif (
                isinstance(self.input_dimension[0], int)
                and self.input_dimension[1] == LayerValidationMethodChoice.minimal
                and len(self.inp_shape[0]) < self.input_dimension[0]
            ):
                return (
                    f"Input dimension Error: Expected dim = {self.input_dimension[0]} or greater "
                    f"but got dim={len(self.inp_shape[0])}!"
                )
            elif (
                isinstance(self.input_dimension[0], tuple)
                and self.input_dimension[1]
                in [
                    LayerValidationMethodChoice.dependence_tuple2,
                    LayerValidationMethodChoice.dependence_tuple3,
                ]
                and len(self.inp_shape[0]) not in self.input_dimension[0]
            ):
                return (
                    f"Input dimension Error: Expected one of {self.input_dimension[0]} "
                    f"input dims but got {len(self.inp_shape[0])}!"
                )
            else:
                return None

    def specific_parameters_validation(self):
        """Validate specific layer parameters or its combination"""

        # initializer identity
        for key in self.layer_parameters.keys():
            if (
                self.layer_parameters.get(key) == "identity"
                and len(self.inp_shape[0]) != 2
            ):
                return (
                    f"Parameters Error: 'Identity' initialazer in {key} can take only 2D input shape "
                    f"but received {len(self.inp_shape[0])}D input shape={self.inp_shape[0]}!"
                )

        # strides and dilation_rate in 1D layers
        if isinstance(self.layer_parameters.get("strides", None), int) and isinstance(
            self.layer_parameters.get("dilation_rate", None), int
        ):
            if (
                self.layer_parameters.get("dilation_rate") > 1
                and self.layer_parameters.get("strides") > 1
            ):
                return "Parameters Error: 'dilation_rate' and 'strides' cannot have value > 1 at the same time!"

        # strides and dilation_rate in 2+D layers
        if isinstance(
            self.layer_parameters.get("strides", None), (tuple, list)
        ) and isinstance(self.layer_parameters.get("strides", None), (tuple, list)):
            if (
                max(self.layer_parameters.get("dilation_rate")) > 1
                and max(self.layer_parameters.get("strides")) > 1
            ):
                return "Parameters Error: 'dilation_rate' and 'strides' cannot have value > 1 at the same time!"

        # value range for axis
        if self.layer_parameters.get("axis", None) and (
            self.layer_parameters.get("axis", None) == 0
            or self.layer_parameters.get("axis", None) > len(self.inp_shape[0]) - 1
            or self.layer_parameters.get("axis", None) < -len(self.inp_shape[0]) + 1
        ):
            axis_values = list(
                np.arange(-len(self.inp_shape[0]) + 1, len(self.inp_shape[0]))
            )
            axis_values.pop(axis_values.index(0))
            return f"Parameters Error: 'axis' can take one of the following values {axis_values}!"

        # groups with data_format, filters and inp_shape
        if (
            self.layer_parameters.get("groups", None)
            and self.layer_parameters.get("data_format", None)
            and self.layer_parameters.get("filters", None)
        ):
            dim = -self.input_dimension[0] + 1
            if self.layer_parameters.get("data_format") == "channels_last" and (
                self.layer_parameters.get("filters")
                % self.layer_parameters.get("groups")
                != 0
                or self.inp_shape[0][-1] % self.layer_parameters.get("groups") != 0
            ):
                return (
                    f"Parameters Error: The number of filters {self.layer_parameters.get('filters')} and "
                    f"channels {self.inp_shape[0][-1]} "
                    f"must be evenly divisible by the number of groups {self.layer_parameters.get('groups')}"
                )

            if self.layer_parameters.get("data_format") == "channels_first" and (
                self.layer_parameters.get("filters")
                % self.layer_parameters.get("groups")
                != 0
                or self.inp_shape[0][dim] % self.layer_parameters.get("groups") != 0
            ):
                return (
                    f"Parameters Error: The number of filters {self.layer_parameters.get('filters')} and "
                    f"channels {self.inp_shape[0][dim]} "
                    f"must be evenly divisible by the number of groups {self.layer_parameters.get('groups')}"
                )

            if (
                self.layer_parameters.get("data_format") == "channels_first"
                and len(self.inp_shape[0]) > -dim + 1
            ):
                if (
                    isinstance(self.layer_parameters.get("strides"), int)
                    and self.layer_parameters.get("strides") > 1
                ) or (
                    isinstance(self.layer_parameters.get("strides"), (tuple, list))
                    and max(self.layer_parameters.get("strides")) > 1
                ):
                    return (
                        f"Parameters Error: for input shape wit dim > {-dim + 1} and 'data_format'='channels_first' "
                        f"parameter 'strides' can not be > 1 but received {self.layer_parameters.get('strides')}"
                    )

        # maxwordcount
        if (
            self.layer_type == LayerTypeChoice.Embedding
            and self.kwargs.get("maxwordcount", None)
            and self.layer_parameters.get("input_dim", None)
        ):
            if self.layer_parameters.get("input_dim") < self.kwargs.get("maxwordcount"):
                return (
                    f"Parameters Error: input_dim={self.layer_parameters.get('input_dim')} must be equal or greater"
                    f"then size of words dictionary (maxwordcount={self.kwargs.get('maxwordcount')})"
                )

        # pretrained models exclusions
        if self.module_type == layers.extra.ModuleTypeChoice.keras_pretrained_model:
            if (
                self.layer_parameters.get("include_top")
                and self.layer_parameters.get("weights")
                and self.layer_parameters.get("classes") != 1000
            ):
                return (
                    f"Parameters Error: If using `weights` as `'imagenet'` with `include_top` as true, "
                    f"`classes` should be 1000 but received {self.layer_parameters.get('classes')}!"
                )
            elif self.layer_type == "InceptionV3":
                if self.layer_parameters.get("include_top") and self.inp_shape[0][
                    1:
                ] != (299, 299, 3):
                    return (
                        f"Input shape Error: with 'include_top'=True input shape "
                        f"must be only (299, 299, 3) but received {self.inp_shape[0][1:]}!"
                    )
                elif (
                    not self.layer_parameters.get("include_top")
                    and self.inp_shape[0][1] < 75
                    or self.inp_shape[0][2] < 75
                    or self.inp_shape[0][3] < 3
                ):
                    return (
                        f"Input shape Error: input shape must be greater or equal (75, 75, 3) "
                        f"in each dim but received input shape {self.inp_shape[0][1:]}!"
                    )
            elif self.layer_type == "Xception":
                if self.layer_parameters.get("include_top") and self.inp_shape[0][
                    1:
                ] != (299, 299, 3):
                    return (
                        f"Input shape Error: with 'include_top'=True input shape "
                        f"must be only (299, 299, 3) but received {self.inp_shape[0][1:]}!"
                    )
                elif (
                    not self.layer_parameters.get("include_top")
                    and self.inp_shape[0][1] < 71
                    or self.inp_shape[0][2] < 71
                    or self.inp_shape[0][3] < 3
                ):
                    return (
                        f"Input shape Error: input shape must be greater or equal (71, 71, 3) "
                        f"in each dim but received input shape {self.inp_shape[0][1:]}!"
                    )
            elif self.layer_type == "VGG16" or self.layer_type == "ResNet50":
                if self.layer_parameters.get("include_top") and self.inp_shape[0][
                    1:
                ] != (224, 224, 3):
                    return (
                        f"Input shape Error: with 'include_top'=True input shape "
                        f"must be only (224, 224, 3) but received {self.inp_shape[0][1:]}!"
                    )
                elif (
                    not self.layer_parameters.get("include_top")
                    and self.inp_shape[0][1] < 32
                    or self.inp_shape[0][2] < 32
                    or self.inp_shape[0][3] < 3
                ):
                    return (
                        f"Input shape Error: input shape must be greater or equal (32, 32, 3) "
                        f"in each dim but received input shape {self.inp_shape[0][1:]}!"
                    )
            else:
                pass

        # CustomUNETBlock exceptions
        if self.layer_type == LayerTypeChoice.CustomUNETBlock:
            if (
                self.inp_shape[0][1] < 32
                or self.inp_shape[0][2] < 32
                or self.inp_shape[0][3] < 3
            ):
                return (
                    f"Input shape Error: input shape must be greater or equal (32, 32, 3) "
                    f"in each dim but received input shape {self.inp_shape[0][1:]}!"
                )
            if self.inp_shape[0][1] % 4 != 0 or self.inp_shape[0][2] % 4 != 0:
                return f"Input shape Error: input shape {self.inp_shape[0]} except channels must be whole divided by 4!"

        # space_to_depth dimentions
        if self.layer_type == LayerTypeChoice.SpaceToDepth:
            if (
                self.layer_parameters.get("data_format")
                == SpaceToDepthDataFormatChoice.NCHW
                or self.layer_parameters.get("data_format")
                == SpaceToDepthDataFormatChoice.NHWC
            ) and len(self.inp_shape[0]) != 4:
                return (
                    f"Input shape Error: expected input shape dim=4 for `data_format`=`NHWC` or `NCHW` but "
                    f"received dim={len(self.inp_shape[0])} with input_shape {self.inp_shape[0]}!"
                )
            if (
                self.layer_parameters.get("data_format")
                == SpaceToDepthDataFormatChoice.NCHW_VECT_C
                and len(self.inp_shape[0]) != 5
            ):
                return (
                    f"Input shape Error: expected input shape dim=5 for `data_format`=`NCHW_VECT_C` but "
                    f"received dim={len(self.inp_shape[0])} with input_shape {self.inp_shape[0]}!"
                )
            if self.layer_parameters.get(
                "data_format"
            ) == SpaceToDepthDataFormatChoice.NCHW_VECT_C and (
                self.inp_shape[0][2] % self.layer_parameters.get("block_size") != 0
                or self.inp_shape[0][3] % self.layer_parameters.get("block_size") != 0
            ):
                return (
                    f"Parameters Error: Dimension size ({self.inp_shape[0][2:4]}) from "
                    f"input_shape {self.inp_shape[0]} both must be evenly divisible by "
                    f"block_size = {self.layer_parameters.get('block_size')}!"
                )


class CustomLayer(tensorflow.keras.layers.Layer):
    """Pattern for create custom user block from block plan"""

    def __init__(self):
        super(CustomLayer, self).__init__()
        self.block_plan = []
        pass

    def __call__(self, input_layer):
        block = None
        for layer in self.block_plan:
            if layer[3] == [-1]:
                block = getattr(self, f"x_{layer[0]}")(input_layer)
                setattr(self, f"out_{layer[0]}", block)
            else:
                if len(layer[3]) == 1:
                    block = getattr(self, f"x_{layer[0]}")(
                        getattr(self, f"out_{layer[3][0]}")
                    )
                    setattr(self, f"out_{layer[0]}", block)
                else:
                    conc_up = []
                    for up in layer[3]:
                        if up == -1:
                            conc_up.append(input_layer)
                        else:
                            conc_up.append(getattr(self, f"out_{up}"))
                    block = getattr(self, f"x_{layer[0]}")(conc_up)
                    setattr(self, f"out_{layer[0]}", block)
        return block


class ModelCreator:
    """Create model from plan object"""

    def __init__(self, terra_model, layer_config):
        super().__init__()
        self.terra_model = terra_model
        self.nnmodel = None
        self.layer_config = layer_config
        # self.debug = False
        self._get_idx_line()
        self._get_model_links()
        self.id_idx_dict = {}
        for _id in self.idx_line:
            for idx in range(len(self.terra_model.plan)):
                if _id == self.terra_model.plan[idx][0]:
                    self.id_idx_dict[_id] = idx
                    break
        self.tensors = {}
        pass

    def _get_model_links(self):
        """Get start_row, uplinks, downlinks from terra_plan"""
        self.start_row, self.uplinks, self.downlinks, _, self.end_row = get_links(
            self.terra_model.plan
        )

    def _get_idx_line(self):
        """Get start_row, uplinks, downlinks from terra_plan"""
        self.idx_line = get_idx_line(self.terra_model.plan)

    def _build_keras_model(self):
        """Build keras model from plan"""
        for _id in self.idx_line:
            layer_type = self.terra_model.plan[self.id_idx_dict.get(_id)][1]
            # if layer_type == 'space_to_depth':  # TODO: костыль для 'space_to_depth'
            #     layer_type = 'SpaceToDepth'
            # module_type = getattr(layers.types, layer_type).LayerConfig.module_type.value
            if (
                self.layer_config.get(_id).module_type.value
                == ModuleTypeChoice.tensorflow
            ):
                self._tf_layer_init(self.terra_model.plan[self.id_idx_dict.get(_id)])
            elif (
                self.layer_config.get(_id).module_type.value
                == ModuleTypeChoice.keras_pretrained_model
            ):
                self._pretrained_model_init_(
                    self.terra_model.plan[self.id_idx_dict.get(_id)]
                )
            elif (
                self.layer_config.get(_id).module_type.value
                == ModuleTypeChoice.block_plan
            ):
                self._custom_block_init(
                    self.terra_model.plan[self.id_idx_dict.get(_id)]
                )
            elif (
                self.layer_config.get(_id).module_type.value == ModuleTypeChoice.keras
                or self.layer_config.get(_id).module_type.value
                == ModuleTypeChoice.terra_layer
            ):
                self._keras_layer_init(self.terra_model.plan[self.id_idx_dict.get(_id)])
            else:
                msg = f'Error: "Layer `{layer_type}` is not found'
                sys.exit(msg)
        inputs = [self.tensors.get(i) for i in self.start_row]
        outputs = [self.tensors.get(i) for i in self.end_row]
        self.nnmodel = tensorflow.keras.Model(inputs, outputs)

    def _keras_layer_init(self, terra_layer):
        """Create keras layer_obj from terra_plan layer"""
        module = importlib.import_module(
            self.layer_config.get(terra_layer[0]).module.value
        )
        if terra_layer[1] == LayerTypeChoice.Input:
            input_shape = self.terra_model.input_shape.get(
                int(terra_layer[2].get("name"))
            )[0]
            self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(
                shape=input_shape, name=terra_layer[2].get("name")
            )
        else:
            if len(terra_layer[3]) == 1:
                input_tensors = self.tensors[terra_layer[3][0]]
            else:
                input_tensors = []
                for idx in terra_layer[3]:
                    input_tensors.append(self.tensors[idx])
            self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(
                **terra_layer[2]
            )(input_tensors)

    def _tf_layer_init(self, terra_layer):
        """Create tensorflow layer_obj from terra_plan layer"""
        module = importlib.import_module(
            self.layer_config.get(terra_layer[0]).module.value
        )

        if len(terra_layer[3]) == 1:
            input_tensors = self.tensors[terra_layer[3][0]]
        else:
            input_tensors = []
            for idx in terra_layer[3]:
                input_tensors.append(self.tensors[idx])
        self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(
            input_tensors, **terra_layer[2]
        )

    def _pretrained_model_init_(self, terra_layer):
        """Create pretrained model as layer_obj from terra_plan layer"""
        module = importlib.import_module(
            self.layer_config.get(terra_layer[0]).module.value
        )
        param2del = ["name", "trainable", "output_layer"]
        attr = copy.deepcopy(terra_layer[2])
        for param in param2del:
            try:
                attr.pop(param)
            except KeyError:
                continue
        layer_object = getattr(module, terra_layer[1])(**attr)

        if terra_layer[2].get("trainable") or terra_layer[2].get("output_layer"):
            for layer in layer_object.layers:
                try:
                    layer.trainable = terra_layer[2].get("trainable")
                except KeyError:
                    continue
            if terra_layer[2].get("output_layer") == "last":
                block_output = layer_object.output
            else:
                block_output = layer_object.get_layer(
                    terra_layer[2].get("output_layer")
                ).output
            layer_object = Model(
                layer_object.input, block_output, name=terra_layer[2].get("name")
            )
        self.tensors[terra_layer[0]] = layer_object(self.tensors[terra_layer[3][0]])

    def _custom_block_init(self, terra_layer):
        block_object = CustomLayer()
        block_object.block_plan = self.terra_model.block_plans.get(terra_layer[0])
        for layer in block_object.block_plan:
            # TODO: поправить на конфиг self.layer_config.get(terra_layer[0]).module.value
            #  когда будет рабочая версия ModelData с блоками
            module = importlib.import_module(
                getattr(layers.types, layer[1]).LayerConfig.module.value
            )
            layer_object = getattr(module, layer[1])(**layer[2])
            setattr(block_object, f"x_{layer[0]}", layer_object)
        # пока реализация с одним входом/выходом
        self.tensors[terra_layer[0]] = block_object(self.tensors[terra_layer[3][0]])

    def create_model(self):
        """Create model from self.model_plan sequences
        Example:
            [(0,"Input",{'name':'input_1},[-1],[1, 2]),
            (1,"Conv2D",{'filters': 32, 'kernel_size': (3, 3)},[0],[2]),
            (2,"Add",{'name': 'output_1'},[0, 1],[]))]

        0 - # layer Index - (int)
        1 - # type of layer - (str)
        2 - # layer parameters - (dict)
        3 - # uplinks - (list of  int)
        4 - # downlinks - (list of int)
        """
        self._build_keras_model()
        return self.nnmodel

    # def compile_model(self, loss=None, optimizer=Adam(), metrics=None):
    #     """Compile tensorflow.keras.Model"""
    #     if metrics is None:
    #         metrics = {'output_1': ["accuracy"]}
    #     if loss is None:
    #         loss = {'output_1': ["categorical_crossentropy"]}
    #     self.nnmodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # def get_model(self) -> Model:
    #     """Get keras.Model"""
    #     return self.nnmodel

    def creator_cleaner(self) -> None:
        """clean and reset to default self.nnmodel"""
        clear_session()
        del self.nnmodel
        gc.collect()
        self.nnmodel = tensorflow.keras.Model()


if __name__ == "__main__":
    input_shape = [
        (None, 100, 100, 28),
        #     # (None, 54, 14, 32),
        #     # (None, 54, 14, 32),
        #     # (None, 54, 14, 32)
    ]
    # input_shape = [TensorShape([None, 64]), TensorShape([None, 64]), TensorShape([None, 64])]
    params = {
        # "size": 2,
        # 'filters': 32,
        # 'kernel_size': (2, 2),
        # 'strides': (3, 3),
        # 'dilation_rate': (1, 1),
        # 'groups': 2,
        # 'depth_multiplier': 5,
        # 'data_format': 'channels_first',
        # "padding": 'valid',
        # "output_padding": None,
        # "kernel_initializer": "glorot_uniform",
        # "beta_initializer": "glorot_uniform",
        # "axis": -1,
        # 'activation': 'relu',
        # "max_value": 1,
        # "rate": 0.2
        # "epsilon": -0.1,
        # "padding": ((1, 0), (-1, 0)),
        # "cropping": ((10, 0), (5, 0)),
        # 'mean': 2.,
        # 'variance': 3.,
        # "target_shape": (54, 168, 32),
        # "input_dim": 2000,
        # "output_dim": 64,
        # 'return_state': True,
        # 'return_sequences': True,
        # 'units': 64,
        # "n": 8,
        # 'include_top': False,
        # 'weights': "imagenet",
        # 'pooling': "avg",
        # "trainable": True,
        # "classes": 1000,
        # "classifier_activation": 'softmax',
        # 'latent_size': 100,
        # 'latent_regularizer': 'vae',
        # 'beta': 5.,
        # 'capacity': 128.,
        # 'randomSample': True,
        # 'roll_up': True,
        "block_size": 2,
        "data_format": SpaceToDepthDataFormatChoice.NCHW,
    }

    # layers.types.Conv2D.LayerConfig.num_uplinks.value = 3
    kwarg = {
        # "maxwordcount": 2000
    }

    layer_name = "space_to_depth"
    # print(get_layer_defaults(layer_name))
    LV = LayerValidation()
    LV.set_state(layer_name, input_shape, params, **kwarg)
    print("\nlayer_type", LV.layer_type)
    print("\nlayer_parameters", LV.layer_parameters)
    print("\nnum_uplinks", LV.num_uplinks)
    print("\ninput_dimension", LV.input_dimension)
    print("\nmodule", LV.module)
    print("\nmodule_type", LV.module_type)
    x, y = LV.get_validated()
    print("\n", x, y)

    x = tensorflow.keras.layers.Input(input_shape[0][1:])
    # x2 = tensorflow.keras.layers.Input(input_shape[1][1:])
    # x3 = tensorflow.keras.layers.Input(input_shape[2][1:])
    # x4 = tensorflow.keras.layers.Input(input_shape[3][1:])
    # x = tensorflow.keras.layers.LSTM(return_state=True, return_sequences=False, units=64)(x)
    # x = getattr(tensorflow.keras.layers, layer_name)(**params)(x)
    # params.pop('trainable')
    # x = getattr(tensorflow.keras.applications.resnet50, layer_name)(**params)(x)
    # x = getattr(tensorflow.keras.layers, layer_name)(**params)([x, x2, x3, x4])
    # x = getattr(customLayers, layer_name)(**params)(x)
    # x = tensorflow_addons.activations.mish(x)
    x = tensorflow.nn.space_to_depth(x, **params)
    print(x.shape)

    # import importlib
    #
    # module = importlib.import_module(def_layer.LayerConfig().module)
    # class_type = getattr(module, layer_type)
    # print(class_type)

    pass
