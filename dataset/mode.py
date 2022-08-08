from enum import IntEnum, auto


class Mode(IntEnum):

    TRAIN = auto()
    VALID = auto()
    TEST = auto()
    INFERENCE = auto()
