import importlib
import sys
import tensorflow
import copy
import gc

from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model

__version__ = 0.080

from tensorflow.python.keras.optimizer_v1 import Adam

from terra_ai.data.modeling import layers
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.modeling.layers.extra import ModuleTypeChoice
# from terra_ai.trgui import get_idx_line, get_links
from terra_ai.validation.validator import get_idx_line, get_links


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
                    block = getattr(self, f"x_{layer[0]}")(getattr(self, f"out_{layer[3][0]}"))
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
    def __init__(self, terra_model):
        super().__init__()
        self.terra_model = terra_model
        self.nnmodel = None
        # self.debug = False
        self._get_idx_line()
        self._get_model_links()
        self.tensors = {}

        pass

    def _get_model_links(self):
        """Get start_row, uplinks, downlinks from terra_plan"""
        self.start_row, self.uplinks, self.downlinks, _, self.end_row = get_links(self.terra_model.plan)

    def _get_idx_line(self):
        """Get start_row, uplinks, downlinks from terra_plan"""
        self.idx_line = get_idx_line(self.terra_model.plan)

    def _build_keras_model(self):
        """Build keras model from plan"""

        for idx in self.idx_line:
            layer_type = self.terra_model.plan[idx][1]
            module_type = getattr(layers.types, layer_type).LayerConfig.module_type.value
            if module_type == ModuleTypeChoice.tensorflow:
                self._tf_layer_init(self.terra_model.plan[idx])
            elif module_type == ModuleTypeChoice.keras_pretrained_model:
                self._pretrained_model_init_(self.terra_model.plan[idx])
            elif module_type == ModuleTypeChoice.block_plan:
                self._custom_block_init(self.terra_model.plan[idx])
            elif module_type == ModuleTypeChoice.keras or module_type == ModuleTypeChoice.terra_layer:
                self._keras_layer_init(self.terra_model.plan[idx])
            else:
                msg = f'Error: "Layer `{layer_type}` is not found'
                sys.exit(msg)
        inputs = [self.tensors.get(i) for i in self.start_row]
        outputs = [self.tensors.get(i) for i in self.end_row]
        self.nnmodel = tensorflow.keras.Model(inputs, outputs)

    def _keras_layer_init(self, terra_layer):
        """Create keras layer_obj from terra_plan layer"""
        module = importlib.import_module(getattr(layers.types, terra_layer[1]).LayerConfig.module.value)
        if terra_layer[1] == LayerTypeChoice.Input:
            input_shape = self.terra_model.input_shape.get(terra_layer[2].get('name'))[0]
            self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(shape=input_shape,
                                                                           name=terra_layer[2].get("name"))
        else:
            if len(terra_layer[3]) == 1:
                input_tensors = self.tensors[terra_layer[3][0]]
            else:
                input_tensors = []
                for idx in terra_layer[3]:
                    input_tensors.append(self.tensors[idx])
            self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(**terra_layer[2])(input_tensors)

    def _tf_layer_init(self, terra_layer):
        """Create tensorflow layer_obj from terra_plan layer"""
        module = importlib.import_module(getattr(layers.types, terra_layer[1]).LayerConfig.module.value)
        if len(terra_layer[3]) == 1:
            input_tensors = self.tensors[terra_layer[3][0]]
        else:
            input_tensors = []
            for idx in terra_layer[3]:
                input_tensors.append(self.tensors[idx])
        self.tensors[terra_layer[0]] = getattr(module, terra_layer[1])(input_tensors, **terra_layer[2])

    def _pretrained_model_init_(self, terra_layer):
        """Create pretrained model as layer_obj from terra_plan layer"""
        module = importlib.import_module(getattr(layers.types, terra_layer[1]).LayerConfig.module.value)
        param2del = ["name", 'trainable', 'output_layer']
        attr = copy.deepcopy(terra_layer[2])
        for param in param2del:
            try:
                attr.pop(param)
            except KeyError:
                continue
        layer_object = getattr(module, terra_layer[1])(**attr)

        if terra_layer[2].get('trainable') or terra_layer[2].get('output_layer'):
            for layer in layer_object.layers:
                try:
                    layer.trainable = terra_layer[2].get('trainable')
                except KeyError:
                    continue
            if terra_layer[2].get('output_layer') == 'last':
                block_output = layer_object.output
            else:
                block_output = layer_object.get_layer(terra_layer[2].get('output_layer')).output
            layer_object = Model(layer_object.input, block_output, name=terra_layer[2].get('name'))
        self.tensors[terra_layer[0]] = layer_object(self.tensors[terra_layer[3][0]])

    def _custom_block_init(self, terra_layer):
        block_object = CustomLayer()
        block_object.block_plan = self.terra_model.block_plans.get(terra_layer[0])
        for layer in block_object.block_plan:
            module = importlib.import_module(getattr(layers.types, layer[1]).LayerConfig.module.value)
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

    def compile_model(self, loss=None, optimizer=Adam(), metrics=None):
        """Compile tensorflow.keras.Model"""
        if metrics is None:
            metrics = {'output_1': ["accuracy"]}
        if loss is None:
            loss = {'output_1': ["categorical_crossentropy"]}
        self.nnmodel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_model(self) -> Model:
        """Get keras.Model"""
        return self.nnmodel

    def creator_cleaner(self) -> None:
        """clean and reset to default self.nnmodel"""
        clear_session()
        del self.nnmodel
        gc.collect()
        self.nnmodel = tensorflow.keras.Model()


if __name__ == '__main__':
    # from terra_ai import trds
    #
    # dataset = trds.DTS()
    # dataset.prepare_dataset(dataset_name='cifar10', source='')
    # x = dataset.X['input_1']['data'][0].reshape((dataset.X['input_1']['data'][0].shape[0], 32 * 32 * 3))[:1000]
    # x = x.reshape((x.shape[0], 32 * 32 * 3))[:1000]
    # dataset.X['input_1']['data'] = (x, x, x)
    # y = dataset.Y['output_1']['data'][0][:1000]
    # dataset.Y['output_1']['data'] = (y, y, y)
    pass