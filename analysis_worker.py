# analysis_worker.py
import os
import collections
from PySide6.QtCore import QObject, Signal


class AnalysisWorker(QObject):
    """在后台线程中执行数据集分析任务。"""
    analysis_complete = Signal(dict)
    log = Signal(str)

    def __init__(self, config, readers_map):
        super().__init__()
        self.config = config
        self.readers_map = readers_map

    def run(self):
        try:
            self.log.emit("分析任务开始...")

            # 1. 使用对应的读取器加载数据
            ReaderClass = self.readers_map[self.config['input_format']]
            reader = ReaderClass(self.config['dataset_path'])
            all_image_annotations, class_names = reader.read()

            if not all_image_annotations:
                self.log.emit("分析中止: 读取器未能加载任何数据。")
                self.analysis_complete.emit(None)  # 发送一个空信号表示失败
                return

            self.log.emit(f"数据加载完成，开始计算统计信息...")

            # 2. 计算统计数据
            class_counts = collections.defaultdict(int)
            image_widths = []
            image_heights = []
            annotations_per_image = []
            bbox_widths_px = []
            bbox_heights_px = []

            for ann_obj in all_image_annotations:
                image_widths.append(ann_obj.width)
                image_heights.append(ann_obj.height)
                annotations_per_image.append(len(ann_obj.annotations))

                for ann in ann_obj.annotations:
                    class_counts[ann.class_id] += 1
                    bbox_widths_px.append(ann.bbox.width * ann_obj.width)
                    bbox_heights_px.append(ann.bbox.height * ann_obj.height)

            # 确保 class_counts 的顺序与 class_names 一致
            ordered_class_counts = [class_counts.get(i, 0) for i in range(len(class_names))]

            # 3. 执行健康检查
            self.log.emit("正在执行健康检查...")
            health_results = self._perform_health_check(all_image_annotations)

            # 4. 准备并发送最终结果
            results = {
                "class_names": class_names,
                "class_counts": ordered_class_counts,
                "image_widths": image_widths,
                "image_heights": image_heights,
                "annotations_per_image": annotations_per_image,
                "bbox_widths_px": bbox_widths_px,
                "bbox_heights_px": bbox_heights_px,
                **health_results  # 合并健康检查结果
            }

            self.log.emit("分析完成！")
            self.analysis_complete.emit(results)

        except Exception as e:
            import traceback
            self.log.emit(f"分析时发生严重错误: {e}\n{traceback.format_exc()}")
            self.analysis_complete.emit(None)

    def _perform_health_check(self, loaded_annotations):
        """
        扫描文件系统以查找不匹配和空文件。
        注意: 这个检查目前主要针对 YOLO/VOC 这种文件结构。
        """
        images_without_labels = []
        labels_without_images = []
        empty_label_files = []

        # 将已加载的数据转换为易于查找的集合
        loaded_image_paths = {ann.image_path for ann in loaded_annotations}

        # 遍历所有可能的 split
        for split in ['train', 'valid', 'test']:
            image_dir = os.path.join(self.config['dataset_path'], split, 'images')
            label_dir = os.path.join(self.config['dataset_path'], split, 'labels')  # 假设是YOLO结构

            if os.path.isdir(image_dir):
                for img_file in os.listdir(image_dir):
                    base_name, _ = os.path.splitext(img_file)
                    # 假设标签文件是 .txt
                    expected_label_path = os.path.join(label_dir, f"{base_name}.txt")
                    if not os.path.exists(expected_label_path):
                        images_without_labels.append(os.path.join(split, 'images', img_file))

            if os.path.isdir(label_dir):
                for lbl_file in os.listdir(label_dir):
                    base_name, _ = os.path.splitext(lbl_file)
                    # 假设图像文件是 .jpg (这是一个简化，可以扩展)
                    expected_image_path = os.path.join(image_dir, f"{base_name}.jpg")
                    if not os.path.exists(expected_image_path) and not os.path.exists(
                            expected_image_path.replace('.jpg', '.png')):
                        labels_without_images.append(os.path.join(split, 'labels', lbl_file))

                    full_label_path = os.path.join(label_dir, lbl_file)
                    if os.path.getsize(full_label_path) == 0:
                        empty_label_files.append(os.path.join(split, 'labels', lbl_file))

        return {
            "images_without_labels": images_without_labels,
            "labels_without_images": labels_without_images,
            "empty_label_files": empty_label_files
        }