# formats/readers/tfrecord_reader.py
import os
import io
from typing import List, Tuple
from .base_reader import BaseReader
from ..internal_data import ImageAnnotation, Annotation, BBox

# --- 可选依赖保护 ---
try:
    import tensorflow as tf
    from PIL import Image

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TfrecordReader(BaseReader):
    """用于读取 TFRecord 数据集格式的读取器。"""

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow 库未安装，无法使用TFRecord读取器。请运行 'pip install tensorflow'。")

    def read(self) -> Tuple[List[ImageAnnotation], List[str]]:
        class_names = self._parse_label_map()
        if not class_names:
            raise FileNotFoundError("找不到 label_map.pbtxt 文件，无法解析类别。")

        class_to_id = {name: i for i, name in enumerate(class_names)}

        record_files = self._find_tfrecord_files()
        if not record_files:
            raise FileNotFoundError("在数据集中未找到任何 .tfrecord 文件。")

        raw_dataset = tf.data.TFRecordDataset(record_files)

        all_annotations = []
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            # 解析 tf.train.Example
            height = example.features.feature['image/height'].int64_list.value[0]
            width = example.features.feature['image/width'].int64_list.value[0]

            # 为图像创建一个唯一标识符，因为没有原始路径
            filename_bytes = example.features.feature['image/filename'].bytes_list.value[0]
            filename = filename_bytes.decode('utf-8')
            image_path_placeholder = os.path.join(self.dataset_path, filename)

            annotations_for_image = []
            xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
            xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
            ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
            ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value
            class_texts = [t.decode('utf-8') for t in
                           example.features.feature['image/object/class/text'].bytes_list.value]

            for i in range(len(xmins)):
                class_id = class_to_id.get(class_texts[i])
                if class_id is None: continue

                # 从归一化的 xmin, ymin, xmax, ymax 转换到我们的标准格式
                w = xmaxs[i] - xmins[i]
                h = ymaxs[i] - ymins[i]
                x_center = xmins[i] + w / 2
                y_center = ymins[i] + h / 2

                bbox = BBox(x_center=x_center, y_center=y_center, width=w, height=h)
                annotations_for_image.append(Annotation(class_id=class_id, bbox=bbox))

            all_annotations.append(ImageAnnotation(
                image_path=image_path_placeholder,
                width=width,
                height=height,
                annotations=annotations_for_image
            ))

        return all_annotations, class_names

    def _find_tfrecord_files(self) -> List[str]:
        return [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.tfrecord')]

    def _parse_label_map(self) -> List[str]:
        label_map_path = os.path.join(self.dataset_path, 'label_map.pbtxt')
        if not os.path.exists(label_map_path):
            return []

        names = []
        with open(label_map_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'name:' in line:
                    name = line.split(':')[-1].strip().replace("'", "").replace('"', '')
                    names.append(name)
        return names