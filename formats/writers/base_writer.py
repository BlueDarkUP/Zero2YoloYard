# formats/writers/base_writer.py
from abc import ABC, abstractmethod
import os
import shutil
import cv2
from ..internal_data import ImageAnnotation

class BaseWriter(ABC):
    def __init__(self, output_dir: str, class_names: list):
        self.output_dir = output_dir
        self.class_names = class_names
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_image(self, image_data_or_path, destination_path):
        if isinstance(image_data_or_path, str):
            shutil.copy2(image_data_or_path, destination_path)
        else:
            image_bgr = cv2.cvtColor(image_data_or_path, cv2.COLOR_RGB2BGR)
            cv2.imwrite(destination_path, image_bgr)

    @abstractmethod
    def write(self, image_data_or_path, image_annotation: ImageAnnotation, new_filename_base: str):
        pass