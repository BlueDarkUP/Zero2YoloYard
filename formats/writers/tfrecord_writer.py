# formats/writers/tfrecord_writer.py
import os
import io
from .base_writer import BaseWriter
from ..internal_data import ImageAnnotation

# --- 可选依赖保护 ---
try:
    import tensorflow as tf
    from PIL import Image

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class TfrecordWriter(BaseWriter):
    """用于写入 TFRecord 数据集格式的写入器。"""

    def __init__(self, output_dir: str, class_names: list):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow 库未安装，无法使用TFRecord写入器。请运行 'pip install tensorflow'。")

        super().__init__(output_dir, class_names)

        # 为一个split创建一个TFRecord文件
        record_filename = os.path.join(self.output_dir, 'output.tfrecord')
        self.writer = tf.io.TFRecordWriter(record_filename)
        self._write_label_map()

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_list_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def write(self, image_data_or_path, image_annotation: ImageAnnotation, new_filename_base: str):
        if isinstance(image_data_or_path, str):
            with tf.io.gfile.GFile(image_data_or_path, 'rb') as fid:
                encoded_image_data = fid.read()
        else:
            # 将numpy数组编码为JPEG
            is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_data_or_path, cv2.COLOR_RGB2BGR))
            encoded_image_data = buffer.tobytes()

        height = image_annotation.height
        width = image_annotation.width
        filename = (new_filename_base + '.jpg').encode('utf-8')

        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        class_texts, class_labels = [], []

        for ann in image_annotation.annotations:
            bbox = ann.bbox
            xmins.append(bbox.x_center - bbox.width / 2)
            xmaxs.append(bbox.x_center + bbox.width / 2)
            ymins.append(bbox.y_center - bbox.height / 2)
            ymaxs.append(bbox.y_center + bbox.height / 2)
            class_texts.append(self.class_names[ann.class_id].encode('utf-8'))
            class_labels.append(ann.class_id)

        feature = {
            'image/height': self._int64_list_feature([height]),
            'image/width': self._int64_list_feature([width]),
            'image/filename': self._bytes_feature(filename),
            'image/source_id': self._bytes_feature(filename),
            'image/encoded': self._bytes_feature(encoded_image_data),
            'image/format': self._bytes_feature(b'jpeg'),
            'image/object/bbox/xmin': self._float_list_feature(xmins),
            'image/object/bbox/xmax': self._float_list_feature(xmaxs),
            'image/object/bbox/ymin': self._float_list_feature(ymins),
            'image/object/bbox/ymax': self._float_list_feature(ymaxs),
            'image/object/class/text': self._bytes_feature(class_texts),
            'image/object/class/label': self._int64_list_feature(class_labels),
        }

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(tf_example.SerializeToString())

    def _write_label_map(self):
        label_map_path = os.path.join(self.output_dir, 'label_map.pbtxt')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            for i, name in enumerate(self.class_names):
                f.write('item {\n')
                f.write(f'  id: {i + 1}\n')
                f.write(f'  name: \'{name}\'\n')
                f.write('}\n\n')

    def finalize(self):
        """关闭TFRecord写入器。"""
        if self.writer:
            self.writer.close()