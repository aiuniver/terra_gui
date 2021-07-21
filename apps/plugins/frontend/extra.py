from enum import Enum


class FieldTypeChoice(str, Enum):
    str = "str"
    int = "int"
    float = "float"
    bool = "bool"
