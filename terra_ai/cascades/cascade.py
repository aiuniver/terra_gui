from typing import Callable
from inspect import signature
from collections import OrderedDict


class Cascade:
    """
    Базовый класс для всех Каскадов

    - Есть имя, которое можно задать
    - Его строковое представление - это его имя (__str__ и __repr__ переопределены для логирования)
    - Этот объект вызываем (обязательно для всех функций и моделей)
    """
    name: str = "Каскад"

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
    out: None

    def __init__(self, fun: Callable, name: str = None):
        super(CascadeElement, self).__init__(name)

        self.fun = fun
        self.input = signature(fun).parameters
        self.output = signature(fun).return_annotation

    def __call__(self, *agr):
        self.out = self.fun(*agr)
        return self.out


class CascadeBlock(Cascade):
    """
    Занимается всей логикой для связи каскадов Может содержать CascadeElement и CascadeBlock Необходима матрица
    смежности (OrderedDict где ключ - Каскад, а значение - list Каскадов, выход из которых пойдёт на вход)

    Если при вызове Каскад передал None, то передаёт None (важно учитывать при создании функций)
    Имя блока - это все имена каскадов, из которых он состоит (у заготовленных блоков __str__ - это заготовленное имя)
    """

    def __init__(self, adjacency_map: OrderedDict):

        self.adjacency_map = adjacency_map
        self.input = signature(next(iter(adjacency_map))).parameters
        self.output = signature(next(reversed(adjacency_map))).return_annotation

        name = "[" + ", ".join([str(x.name) for x in adjacency_map]) + "]"
        super(CascadeBlock, self).__init__(name)

    def __call__(self, item):
        self.out_map = {}

        global cascade
        for cascade, inp in self.adjacency_map.items():
            if cascade(*[item if j == 'INPUT' else j.out for j in inp]) is None:
                return None

        self.out = cascade.out
        return cascade.out

    def loop(self, iterator):
        for item in iterator:
            yield self.__call__(item)


class BuildModelCascade(CascadeBlock):

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
