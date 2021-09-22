from typing import Callable
from collections import OrderedDict


class Cascade:
    """
    Базовый класс для всех Каскадов

    - Есть имя, которое можно задать
    - Его строковое представление - это его имя (__str__ и __repr__ переопределены для логирования)
    - Этот объект вызываем (обязательно для всех функций и моделей)
    """
    name: str = "Каскад"
    out = None

    def __init__(self, name: str):
        self.name = name if name is not None else self.name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


class CascadeElement(Cascade):
    """
    Используются для вызова функции или модели
    """

    def __init__(self, fun: Callable, name: str = None):
        super(CascadeElement, self).__init__(name)

        self.fun = fun

    def __call__(self, *agr):
        self.out = self.fun(*agr)
        return self.out


class CascadeOutput(Cascade):
    def __init__(self, iter: Callable, params: dict, name: str = "Recorder"):
        super(CascadeOutput, self).__init__(name)
        self.iter = iter
        self.params = params

        self.recorder = self.writer = None

    def choose_path(self, path: str):
        out = self.iter(path, **self.params)

        if isinstance(out, tuple):
            self.writer, self.recorder = out[0], out[1]
        else:
            self.recorder = out

    def release(self):
        if self.writer:
            self.writer.release()

    def __call__(self, img):
        self.recorder(img)


class CascadeBlock(Cascade):
    """
    Занимается всей логикой для связи каскадов Может содержать CascadeElement и CascadeBlock Необходима матрица
    смежности (OrderedDict где ключ - Каскад, а значение - list Каскадов, выход из которых пойдёт на вход)

    Если при вызове Каскад передал None, то передаёт None (важно учитывать при создании функций)
    Имя блока - это все имена каскадов, из которых он состоит (у заготовленных блоков __str__ - это заготовленное имя)
    """

    def __init__(self, adjacency_map: OrderedDict):

        self.adjacency_map = adjacency_map
        self.cascades = list(adjacency_map.keys())
        name = "[" + ", ".join([str(x.name) for x in adjacency_map]) + "]"
        super(CascadeBlock, self).__init__(name)

    def __getitem__(self, index):
        return self.cascades[index]

    def __call__(self, item):
        self.out_map = {}

        global cascade
        for cascade, inp in self.adjacency_map.items():
            cascade_input = []
            for j in inp:
                j = item if j == 'INPUT' else j.out

                if j is None:
                    return j

                cascade_input.append(j)

            cascade(*cascade_input)

        self.out = cascade.out
        return cascade.out


class CompleteCascade(Cascade):
    def __init__(self, input_cascade: Callable, adjacency_map: OrderedDict):

        self.input = input_cascade
        self.output = []

        for i in adjacency_map:
            if isinstance(i, CascadeOutput):
                self.output.append(i)

        self.cascade_block = CascadeBlock(adjacency_map)
        super(CompleteCascade, self).__init__(self.cascade_block.name)

    def __getitem__(self, index):  # пока не ясно, стоит ли наследовать от CascadeBlock
        return self.cascade_block[index]

    def __call__(self, input_path, output_path):

        if len(self.output) == 1:
            self.output[0].choose_path(output_path)
        elif len(self.output) > 1:
            for i, out in enumerate(self.output):
                path = output_path[:-4] + f"_{i}" + output_path[-4:]
                out.choose_path(path)

        for img in self.input(input_path):
            self.cascade_block(img)

        for i in self.output:
            i.release()


class BuildModelCascade(CascadeBlock):
    """
    Класс который вызывается при создании модели из json
    """
    def __init__(self, preprocess, model, postprocessing, name=None):

        self.preprocess = CascadeElement(preprocess, name="Препроцесс") if preprocess else preprocess
        self.model = CascadeElement(model, name=name) if name else CascadeElement(model, name="Модель")
        self.postprocessing = CascadeElement(postprocessing, name="Постпроцесс") if postprocessing else postprocessing

        adjacency_map = OrderedDict()
        if self.preprocess:
            adjacency_map[self.preprocess] = ['INPUT']

        adjacency_map[self.model] = [self.preprocess]

        if self.postprocessing:
            adjacency_map[self.postprocessing] = [self.model, 'INPUT']

        super(BuildModelCascade, self).__init__(adjacency_map)
