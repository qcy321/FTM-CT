from enum import Enum

from pandas import DataFrame


class Classes(Enum):
    TEST = "test"
    TRAIN = "train"
    VALID = "valid"


class DfData:
    """
    用于处理数据，承载数据的类
    """

    def __init__(self, df: DataFrame, file: str):
        """
        初始化
        :param df: df数据，存储df数据
        :param file: df数据以某种格式存储在文件中的路径，即文件存储路径
        """
        self.df = df
        self.file = file

    def __str__(self):
        return f"df: {self.df}, file: {self.file}"


class OpcodeData:

    def __init__(self) -> None:
        self.func_name = ""
        self.docstring = ""
        self.code = ""
        self.opcode = ""
        self.line = []
