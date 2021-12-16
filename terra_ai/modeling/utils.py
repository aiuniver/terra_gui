import copy
from typing import List, Tuple, Optional
from tensorflow import TensorShape
from terra_ai.data.modeling.extra import LayerGroupChoice, LayerTypeChoice
from terra_ai.data.modeling.layer import LayerData


def get_links(model_plan: List[tuple]) -> Tuple[list, dict, dict, list, list]:
    # logger.debug(f"Validator module, {get_links.__name__}")
    start_row = []
    end_row = []
    up_links = {}
    down_links = {}
    all_indexes = []
    for layer in model_plan:
        if layer[3] == [-1]:
            start_row.append(layer[0])
        if not layer[4]:
            end_row.append(layer[0])
        all_indexes.append(layer[0])
        down_links[layer[0]] = layer[4]
        up_links[layer[0]] = layer[3]
    return start_row, up_links, down_links, all_indexes, end_row


def get_idx_line(model_plan: List[tuple]):
    # logger.debug(f"Validator module, {get_idx_line.__name__}")
    start_row, up_links, down_links, idx2remove, _ = get_links(model_plan)
    distribution = []  # distribution plan, show rows with layers

    for i in start_row:
        if up_links[i] != [-1]:
            start_row.pop(start_row.index(i))

    for i in start_row:
        idx2remove.pop(idx2remove.index(i))
    distribution.append(start_row)

    # get other rows
    count = 1
    while idx2remove:
        count += 1
        row_idx_s = []
        for idx in distribution[-1]:
            for down_link in down_links.get(idx):
                if down_link not in row_idx_s:
                    row_idx_s.append(down_link)

        row_idx_s_copy = copy.deepcopy(row_idx_s)
        for link in row_idx_s:
            if (
                    len(up_links.get(link)) > 1
                    and len(set(idx2remove) & set(up_links.get(link))) != 0
            ):
                row_idx_s_copy.pop(row_idx_s_copy.index(link))
        row_idx_s = row_idx_s_copy

        distribution.append(row_idx_s)
        for idx in row_idx_s:
            idx2remove.pop(idx2remove.index(idx))
    idx_line = []
    for row in distribution:
        idx_line.extend(row)
    return idx_line


def reorder_plan(model_plan: List[tuple]):
    # logger.debug(f"Validator module, {reorder_plan.__name__}")
    idx_line = get_idx_line(model_plan)
    order_plan = []
    for idx in idx_line:
        for layer in model_plan:
            if idx == layer[0]:
                order_plan.append(layer)
                break
    return order_plan


def get_edges(model_plan: List[tuple], full_connection: bool = False) -> List[Tuple[int, int]]:
    # logger.debug(f"Validator module, {get_edges.__name__}")
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


def reformat_input_shape(input_sh: List[Tuple[Optional[int]]]) -> List[Tuple[Optional[int]]]:
    # logger.debug(f"Validator module, {reformat_input_shape.__name__}")
    if len(input_sh) == 1:
        if input_sh[0][0]:
            input_sh = list(input_sh[0])
            input_sh.insert(0, None)
            return [tuple(input_sh)]
        else:
            return input_sh
    else:
        new_input = []
        for inp in input_sh:
            if inp[0]:
                inp = list(inp)
                inp.insert(0, None)
                new_input.append(tuple(inp))
            else:
                new_input.append(inp)
        return new_input


def get_layer_info(layer_strict: LayerData, block_name=None) -> tuple:
    # logger.debug(f"Validator module, {get_layer_info.__name__}")
    params_dict = layer_strict.parameters.merged
    if layer_strict.type == LayerTypeChoice.PretrainedYOLO:
        print('layer_strict.parameters.weight_path', str(layer_strict.parameters.weight_path))
        params_dict['save_weights'] = layer_strict.parameters.weight_path
    if layer_strict.group == LayerGroupChoice.input or layer_strict.group == LayerGroupChoice.output:
        params_dict["name"] = f"{layer_strict.id}"
    elif block_name:
        params_dict["name"] = f"{block_name}_{layer_strict.name}"
    else:
        params_dict["name"] = f"{layer_strict.type}_{layer_strict.id}"
    return (
        layer_strict.id,
        layer_strict.type.value,
        params_dict,
        [-1] if not layer_strict.bind.up else [-1 if i is None else i for i in layer_strict.bind.up],
        [i for i in layer_strict.bind.down],
    )


def tensor_shape_to_tuple(tensor_shape: TensorShape):
    # logger.debug(f"Validator module, {tensor_shape_to_tuple.__name__}")
    tuple_shape = []
    for dim in tensor_shape:
        tuple_shape.append(dim)
    return tuple(tuple_shape)
