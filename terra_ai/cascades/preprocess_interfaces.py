from abc import ABC, abstractmethod

from terra_ai.datasets.arrays_create import CreateArray


class Preprocess(ABC):

    def __init__(self):
        self.preprocess = CreateArray()

    @abstractmethod
    def execute(self, sources, **options):
        pass


class ImagePreprocess(Preprocess):

    def __init__(self):
        super(ImagePreprocess, self).__init__()

    def execute(self, sources, **options):
        instructions = self.preprocess.create_image(sources, **options)
        cut = self.preprocess.cut_image()
