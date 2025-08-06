import os
import shutil
import yaml
import time
import random
from PySide6.QtCore import QObject, Signal, QThread

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


    class YOLO:
        pass

from formats.internal_data import ImageAnnotation


class TrainingWorker(QObject):
    finished = Signal(str)
    progress = Signal(int, str)
    log = Signal(str)
    error = Signal(str)

    def __init__(self, image_annotations, class_names, train_params):
        super().__init__()
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics is not installed. Please run 'pip install ultralytics'.")

        self.image_annotations = image_annotations
        self.class_names = class_names # This is the crucial part: use the class_names passed in
        self.train_params = train_params
        self.is_cancelled = False
        self.yolo_model = None

    def _prepare_split_data(self, annotations, split_name, temp_dataset_dir, progress_start, progress_range):
        temp_images_dir = os.path.join(temp_dataset_dir, "images", split_name)
        temp_labels_dir = os.path.join(temp_dataset_dir, "labels", split_name)
        os.makedirs(temp_images_dir, exist_ok=True)
        os.makedirs(temp_labels_dir, exist_ok=True)

        total_files = len(annotations)
        for i, ann_obj in enumerate(annotations):
            if self.is_cancelled:
                return False

            progress_val = progress_start + int((i / total_files) * progress_range)
            self.progress.emit(progress_val, f"准备 {split_name} 数据: {i + 1}/{total_files}")

            base_name = os.path.basename(ann_obj.image_path)
            dest_img_path = os.path.join(temp_images_dir, base_name)
            try:
                shutil.copy2(ann_obj.image_path, dest_img_path)
            except FileNotFoundError:
                self.log.emit(f"警告: 找不到源文件 {ann_obj.image_path}，跳过。")
                continue

            label_name = f"{os.path.splitext(base_name)[0]}.txt"
            dest_label_path = os.path.join(temp_labels_dir, label_name)
            with open(dest_label_path, 'w', encoding='utf-8') as f:
                for ann in ann_obj.annotations:
                    line = f"{ann.class_id} {ann.bbox.x_center:.6f} {ann.bbox.y_center:.6f} {ann.bbox.width:.6f} {ann.bbox.height:.6f}\n"
                    f.write(line)
        return True

    def run(self):
        temp_dataset_dir = None
        try:
            self.log.emit("创建临时训练目录...")
            temp_base_dir = "temp_training"
            run_name = f"run_{int(time.time())}"
            temp_dataset_dir = os.path.join(temp_base_dir, run_name)
            os.makedirs(temp_dataset_dir, exist_ok=True)
            self.log.emit(f"临时目录已创建: {temp_dataset_dir}")

            val_split_ratio = self.train_params.get('val_split_ratio', 0.2)
            all_annotations = list(self.image_annotations)
            random.shuffle(all_annotations)

            val_size = int(len(all_annotations) * val_split_ratio)
            train_annotations = all_annotations[val_size:]
            val_annotations = all_annotations[:val_size]

            if not train_annotations:
                self.error.emit("错误：没有可用于训练的数据。")
                return

            self.log.emit(f"数据分割: {len(train_annotations)} 训练, {len(val_annotations)} 验证。")

            if not self._prepare_split_data(train_annotations, 'train', temp_dataset_dir, 0, 40):
                self.log.emit("任务被取消。")
                self._cleanup(temp_dataset_dir)
                return

            if val_annotations:
                if not self._prepare_split_data(val_annotations, 'val', temp_dataset_dir, 40, 10):
                    self.log.emit("任务被取消。")
                    self._cleanup(temp_dataset_dir)
                    return

            self.log.emit("数据准备完成。")
            self.progress.emit(50, "数据准备完成。")

            self.log.emit("创建 data.yaml 文件...")

            val_path = os.path.join('images', 'val') if val_annotations else os.path.join('images', 'train')
            if not val_annotations:
                self.log.emit("警告: 验证集为空，将使用训练集进行验证。这可能导致评估指标偏高。")

            data_yaml_content = {
                'path': os.path.abspath(temp_dataset_dir),
                'train': os.path.join('images', 'train'),
                'val': val_path,
                'names': {i: name for i, name in enumerate(self.class_names)} # Use self.class_names directly
            }
            data_yaml_path = os.path.join(temp_dataset_dir, 'data.yaml')
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml_content, f, allow_unicode=True, default_flow_style=False)

            self.log.emit("开始 YOLOv8 训练...")
            self.progress.emit(50, "正在启动训练...")

            self.yolo_model = YOLO(self.train_params['base_model'])

            self.log.emit("注意：训练开始后无法中途取消。")

            results = self.yolo_model.train(
                data=data_yaml_path,
                epochs=self.train_params['epochs'],
                batch=self.train_params['batch_size'],
                imgsz=self.train_params['img_size'],
                project='runs/train',
                name=run_name,
                device=self.train_params.get('device', 'cpu'),
                workers=self.train_params.get('workers', 0),
                patience=self.train_params.get('patience', 50)
            )

            self.progress.emit(100, "训练完成。")
            self.log.emit("训练成功完成！")

            best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
            self.log.emit(f"最佳模型已保存至: {best_model_path}")

            self.finished.emit(str(best_model_path))

        except Exception as e:
            import traceback
            self.error.emit(f"训练时发生严重错误: {e}\n{traceback.format_exc()}")
        finally:
            if temp_dataset_dir and os.path.exists(temp_dataset_dir):
                self.log.emit(f"保留临时训练数据于: {temp_dataset_dir}")

    def _cleanup(self, directory):
        if directory and os.path.exists(directory):
            try:
                shutil.rmtree(directory)
                self.log.emit(f"临时目录 {directory} 已清理。")
            except Exception as e:
                self.log.emit(f"清理临时目录 {directory} 失败: {e}")

    def cancel(self):
        self.is_cancelled = True
        self.log.emit("收到取消请求...")