# formats/readers/base_reader.py
from abc import ABC, abstractmethod
from typing import List, Tuple
from ..internal_data import ImageAnnotation

class BaseReader(ABC):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    @abstractmethod
    def read(self) -> Tuple[List[ImageAnnotation], List[str]]:
        """
        读取整个数据集并将其转换为标准内部格式。

        Returns:
            A tuple containing:
            - list: 一个包含所有 ImageAnnotation 对象的列表。
            - list: 一个包含所有类别名称的字符串列表。
        """
        pass