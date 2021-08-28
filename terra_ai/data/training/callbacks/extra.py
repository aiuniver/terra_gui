from enum import Enum


class ShowImagesChoice(str, Enum):
    Best = "Best"
    Worst = "Worst"
    Random = "Random"
    Seed = "Seed"
