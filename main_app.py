import subprocess
import sys
import os
import json
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLineEdit, QFileDialog,
                               QCheckBox, QSlider, QLabel, QProgressBar, QGroupBox,
                               QFormLayout, QTextEdit, QScrollArea, QSpinBox, QMessageBox,
                               QComboBox, QInputDialog, QListWidget, QDialog, QListWidgetItem, QProgressDialog)
from PySide6.QtCore import Qt, QThread, QEvent
from PySide6.QtGui import QCursor, QIcon

from analysis_worker import AnalysisWorker
from dashboard_dialog import DashboardDialog
from annotation_window import AnnotationWindow
from split_dialog import SplitDialog
from split_worker import SplitWorker
from video_processor_dialog import VideoProcessorDialog

# --- 核心修改部分开始 ---
# 分别检查 TensorFlow 和 AI 模型的可用性

# 检查 TensorFlow
try:
    import tensorflow
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# 检查 AI 模型 (PyTorch, Ultralytics, SAM)
try:
    from ai_models import AI_AVAILABLE
except ImportError:
    AI_AVAILABLE = False
# --- 核心修改部分结束 ---


from augmentations_config import AUGMENTATIONS
from worker import AugmentationWorker
from formats.detector import detect_format
from formats.readers.yolo_reader import YoloReader
from formats.readers.voc_reader import VocReader
from formats.readers.coco_reader import CocoReader
from formats.internal_data import ImageAnnotation
from formats.writers.yolo_writer import YoloWriter
from formats.writers.voc_writer import VocWriter
from formats.writers.coco_writer import CocoWriter

if TENSORFLOW_AVAILABLE:
    from formats.readers.tfrecord_reader import TfrecordReader
    from formats.writers.tfrecord_writer import TfrecordWriter


class NoWheelQSlider(QSlider):
    def __init__(self, orientation, parent=None): super().__init__(orientation, parent); self.wheelEvent = lambda \
            e: e.ignore()


class NoWheelQSpinBox(QSpinBox):
    def __init__(self, parent=None): super().__init__(parent); self.wheelEvent = lambda e: e.ignore()


class AugmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Zero-to YOLO Yard IDE")
        self.setGeometry(100, 100, 1100, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_v_layout = QVBoxLayout(self.central_widget)

        top_panel = QWidget()
        top_panel_layout = QHBoxLayout(top_panel)
        self.left_panel = QWidget()
        self.right_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.right_layout = QVBoxLayout(self.right_panel)
        top_panel_layout.addWidget(self.left_panel, 2)
        top_panel_layout.addWidget(self.right_panel, 3)
        self.main_v_layout.addWidget(top_panel)

        self.readers_map = {"YOLO": YoloReader, "Pascal VOC": VocReader, "COCO": CocoReader}
        self.writers_map = {"YOLOv5 PyTorch": YoloWriter, "Pascal VOC": VocWriter, "COCO": CocoWriter}
        if TENSORFLOW_AVAILABLE:
            self.readers_map["TFRecord"] = TfrecordReader
            self.writers_map["TFRecord"] = TfrecordWriter

        self.controls = {}
        self.worker = None
        self.thread = None
        self.analysis_worker = None
        self.analysis_thread = None
        self.split_worker = None
        self.split_thread = None
        self.all_image_annotations = []
        self.all_video_paths = []

        self.presets_dir = "presets"
        os.makedirs(self.presets_dir, exist_ok=True)

        self.create_project_widgets(self.left_layout)
        self.create_file_list_widget(self.left_layout)
        self.create_settings_widgets(self.right_layout)
        self.create_adv_gen_widgets(self.right_layout)
        self.create_augmentation_widgets(self.right_layout)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.create_control_widgets(bottom_layout)
        self.main_v_layout.addWidget(bottom_widget)

        self.load_stylesheet()

    def load_stylesheet(self):
        try:
            with open("style.qss", "r", encoding='utf-8') as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            print("警告: style.qss 未找到。")

    def create_project_widgets(self, parent_layout):
        group = QGroupBox("项目与工具")
        layout = QFormLayout()
        tools_layout = QHBoxLayout()
        split_btn = QPushButton("🪓 数据集分割工具")
        split_btn.clicked.connect(self.open_split_tool)
        tools_layout.addWidget(split_btn)
        tools_layout.addStretch()
        layout.addRow(tools_layout)
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择数据集根目录")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self.browse_dataset_path)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(browse_btn)
        layout.addRow("数据集路径:", path_layout)
        interactive_tools_layout = QHBoxLayout()
        self.detected_format_label = QLabel("未检测")
        self.detected_format_label.setStyleSheet("font-weight: bold; color: #f0f0f0;")
        self.analyze_btn = QPushButton("📊 分析")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.annotate_btn = QPushButton("✏️ 标注")
        self.annotate_btn.clicked.connect(self.open_annotation_window)
        self.analyze_btn.setEnabled(False)
        self.annotate_btn.setEnabled(False)
        interactive_tools_layout.addWidget(QLabel("检测格式:"))
        interactive_tools_layout.addWidget(self.detected_format_label, 1)
        interactive_tools_layout.addWidget(self.analyze_btn)
        interactive_tools_layout.addWidget(self.annotate_btn)
        layout.addRow(interactive_tools_layout)
        self.format_combo = QComboBox()
        self.format_combo.addItems(self.writers_map.keys())
        layout.addRow("目标导出格式:", self.format_combo)
        if not TENSORFLOW_AVAILABLE:
            tf_label = QLabel("TFRecord 支持已禁用")
            tf_label.setStyleSheet("color: #aaa;")
            layout.addRow(tf_label)
        if not AI_AVAILABLE:
            ai_label = QLabel("AI 辅助功能已禁用 (缺少依赖)")
            ai_label.setStyleSheet("color: #aaa;")
            layout.addRow(ai_label)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_file_list_widget(self, parent_layout):
        group = QGroupBox("文件浏览器")
        layout = QVBoxLayout(group)
        self.file_list_widget = QListWidget()
        self.file_list_widget.doubleClicked.connect(self.on_file_double_clicked)
        reset_filter_btn = QPushButton("🔄 重置筛选")
        reset_filter_btn.clicked.connect(lambda: self.populate_file_list())
        layout.addWidget(self.file_list_widget)
        layout.addWidget(reset_filter_btn)
        parent_layout.addWidget(group, 1)

    def create_settings_widgets(self, parent_layout):
        group = QGroupBox("性能与通用设置")
        layout = QFormLayout()
        try:
            num_cores = os.cpu_count()
        except:
            num_cores = 4
        self.worker_spin = NoWheelQSpinBox()
        self.worker_spin.setRange(1, num_cores * 2)
        self.worker_spin.setValue(num_cores)
        layout.addRow("并行处理核心数:", self.worker_spin)
        self.preload_cb = QCheckBox("预加载图像到内存 (更快，但消耗更多RAM)")
        self.preload_cb.setChecked(False)
        self.preload_cb.stateChanged.connect(self.on_preload_toggled)
        layout.addRow(self.preload_cb)
        self.num_augs_spin = NoWheelQSpinBox()
        self.num_augs_spin.setRange(0, 50)
        self.num_augs_spin.setValue(4)
        layout.addRow("标准增强数量:", self.num_augs_spin)
        self.copy_originals_cb = QCheckBox("同时导出原始图像")
        self.copy_originals_cb.setChecked(True)
        layout.addRow(self.copy_originals_cb)
        presets_layout = QHBoxLayout()
        save_preset_btn = QPushButton("保存预设...")
        save_preset_btn.clicked.connect(self.save_preset)
        load_preset_btn = QPushButton("加载预设...")
        load_preset_btn.clicked.connect(self.load_preset)
        presets_layout.addWidget(save_preset_btn)
        presets_layout.addWidget(load_preset_btn)
        layout.addRow("配置预设:", presets_layout)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def on_preload_toggled(self, state):
        if state == Qt.Checked:
            if not self.all_image_annotations:
                QMessageBox.information(self, "提示", "请先选择一个数据集，再启用此选项。")
                self.preload_cb.setChecked(False)
                return

            num_images = len(self.all_image_annotations)
            avg_width = np.mean([ann.width for ann in self.all_image_annotations if ann.width > 0])
            avg_height = np.mean([ann.height for ann in self.all_image_annotations if ann.height > 0])
            if np.isnan(avg_width) or avg_width == 0: avg_width = 640
            if np.isnan(avg_height) or avg_height == 0: avg_height = 640

            estimated_mb = (num_images * avg_width * avg_height * 3) / (1024 * 1024)

            reply = QMessageBox.question(self, "内存警告",
                                         f"您即将预加载 {num_images} 张图片。\n"
                                         f"预计将消耗大约 **{estimated_mb:.0f} MB** 的内存。\n\n"
                                         "如果您的可用内存不足，程序可能会变得非常缓慢或崩溃。\n"
                                         "您确定要继续吗？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.No:
                self.preload_cb.setChecked(False)

    def create_adv_gen_widgets(self, parent_layout):
        group = QGroupBox("高级复合增强")
        layout = QFormLayout()
        self.mosaic_cb = QCheckBox("启用 Mosaic 增强")
        self.num_mosaic_spin = NoWheelQSpinBox()
        self.num_mosaic_spin.setRange(1, 10000)
        self.num_mosaic_spin.setValue(100)
        layout.addRow(self.mosaic_cb)
        layout.addRow("Mosaic图像数量:", self.num_mosaic_spin)
        group.setLayout(layout)
        parent_layout.addWidget(group)

    def create_augmentation_widgets(self, parent_layout):
        group = QGroupBox("标准单图增强")
        group_layout = QVBoxLayout(group)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        group_layout.addWidget(scroll_area)
        for category, augs in AUGMENTATIONS.items():
            cat_group = QGroupBox(category)
            cat_layout = QFormLayout()
            for aug_name, config in augs.items():
                cb = QCheckBox(aug_name)
                cat_layout.addRow(cb)
                self.controls[aug_name] = {'checkbox': cb, 'config': config, 'params': {}}
                param_container = QWidget()
                param_layout = QFormLayout(param_container)
                param_layout.setContentsMargins(20, 5, 5, 5)
                for param_name, p_config in config['params'].items():
                    label = QLabel(f"{p_config['label']}:")
                    if p_config['type'] == 'float':
                        slider = NoWheelQSlider(Qt.Horizontal)
                        slider.setRange(0, 100)
                        slider.setValue(int(p_config['default'] * 100))
                        val_label = QLabel(f"{p_config['default']:.2f}")
                        slider.valueChanged.connect(lambda v, lbl=val_label, m=100.0: lbl.setText(f"{v / m:.2f}"))
                        h_layout = QHBoxLayout()
                        h_layout.addWidget(slider)
                        h_layout.addWidget(val_label, 0, Qt.AlignRight)
                        param_layout.addRow(label, h_layout)
                        self.controls[aug_name]['params'][param_name] = slider
                    elif p_config['type'] == 'range':
                        min_abs, max_abs = abs(p_config['range'][0]), abs(p_config['range'][1])
                        slider = NoWheelQSlider(Qt.Horizontal)
                        slider.setRange(0, max_abs)
                        slider.setValue(abs(p_config['default'][1]))
                        val_label = QLabel(f"±{p_config['default'][1]}")
                        slider.valueChanged.connect(lambda v, lbl=val_label: lbl.setText(f"±{v}"))
                        h_layout = QHBoxLayout()
                        h_layout.addWidget(slider)
                        h_layout.addWidget(val_label, 0, Qt.AlignRight)
                        param_layout.addRow(label, h_layout)
                        self.controls[aug_name]['params'][param_name] = slider
                    elif p_config['type'] in ['int', 'int_static']:
                        if p_config['type'] == 'int':
                            spin = NoWheelQSpinBox()
                            spin.setRange(p_config['range'][0], p_config['range'][1])
                            spin.setValue(p_config['default'])
                            param_layout.addRow(label, spin)
                            self.controls[aug_name]['params'][param_name] = spin
                        else:
                            static_label = QLabel(str(p_config['default']))
                            param_layout.addRow(label, static_label)
                            self.controls[aug_name]['params'][param_name] = static_label
                param_container.setLayout(param_layout)
                cat_layout.addRow(param_container)
            cat_group.setLayout(cat_layout)
            scroll_layout.addWidget(cat_group)
        scroll_area.setWidget(scroll_widget)
        parent_layout.addWidget(group, 1)

    def create_control_widgets(self, parent_layout):
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 开始处理")
        self.start_btn.clicked.connect(self.start_augmentation)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_augmentation)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        parent_layout.addLayout(control_layout)
        self.progress_bar = QProgressBar()
        parent_layout.addWidget(self.progress_bar)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(150)
        parent_layout.addWidget(self.log_output)

    def gather_config(self, for_analysis=False):
        detected_format_text = self.detected_format_label.text()
        detected_format = detected_format_text.split('(')[0].strip()

        if detected_format_text.startswith("新数据集") or detected_format_text.startswith("未识别格式 (包含视频)"):
            detected_format = "YOLO"
            config = {"dataset_path": self.path_edit.text(), "input_format": detected_format, "is_new_dataset": True}
            os.makedirs(os.path.join(self.path_edit.text(), 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.path_edit.text(), 'labels'), exist_ok=True)
        elif detected_format in ["Unknown", "未检测"]:
            QMessageBox.warning(self, "错误", "未选择或无法识别数据集！")
            return None
        else:
            if detected_format not in self.readers_map:
                QMessageBox.warning(self, "错误", f"不支持的格式: {detected_format} 或需要 TensorFlow。")
                return None
            config = {"dataset_path": self.path_edit.text(), "input_format": detected_format}

        if not for_analysis:
            output_format = self.format_combo.currentText()
            if not self.writers_map.get(output_format):
                if output_format == "TFRecord" and not TENSORFLOW_AVAILABLE:
                    QMessageBox.warning(self, "错误", f"写入器 for '{output_format}' 需要 TensorFlow。")
                    return None
                QMessageBox.warning(self, "错误", f"无法找到 '{output_format}' 的写入器。")
                return None


            config.update({
                "output_format": output_format, "num_augs": self.num_augs_spin.value(),
                "copy_originals": self.copy_originals_cb.isChecked(),
                "mosaic_enabled": self.mosaic_cb.isChecked(), "num_mosaic": self.num_mosaic_spin.value(),
                "num_workers": self.worker_spin.value(), "preload_images": self.preload_cb.isChecked(),
                "augmentations": {}
            })
            for aug_name, ctrl in self.controls.items():
                if ctrl['checkbox'].isChecked():
                    params_values = {}
                    for param_name, widget in ctrl['params'].items():
                        p_config = ctrl['config']['params'][param_name]
                        if p_config['type'] == 'range':
                            val = widget.value()
                            params_values[param_name] = (-val, val)
                        elif isinstance(widget, QSlider):
                            params_values[param_name] = widget.value() / 100.0
                        elif isinstance(widget, QSpinBox):
                            params_values[param_name] = widget.value()
                        elif isinstance(widget, QLabel):
                            params_values[param_name] = int(widget.text())

                    config['augmentations'][aug_name] = {"enabled": True, "class": ctrl['config']['class'],
                                                         "values": params_values, "config": ctrl['config']}
        return config

    def browse_dataset_path(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据集根目录或包含图片/视频的目录")
        if path:
            self.path_edit.setText(path)

            detected = detect_format(path)
            self.all_video_paths.clear()

            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
            has_videos_in_browsed_path = False

            check_dirs = [path]
            if os.path.isdir(os.path.join(path, 'images')):
                check_dirs.append(os.path.join(path, 'images'))

            for check_dir in check_dirs:
                if not os.path.isdir(check_dir):
                    continue
                for filename in os.listdir(check_dir):
                    file_path = os.path.join(check_dir, filename)
                    if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in video_extensions:
                        self.all_video_paths.append(file_path)
                        has_videos_in_browsed_path = True

            self.analyze_btn.setEnabled(False)
            self.annotate_btn.setEnabled(False)

            if detected == "New Dataset":
                self.detected_format_label.setText("新数据集 (空白标注)")
                self.detected_format_label.setStyleSheet("font-weight: bold; color: #55aaff;")
                self.annotate_btn.setEnabled(True)
            elif detected == "TFRecord" and not TENSORFLOW_AVAILABLE:
                self.detected_format_label.setText(f"{detected} (需要 TensorFlow)")
                self.detected_format_label.setStyleSheet("font-weight: bold; color: #f39c12;")
            elif detected == "Unknown" and has_videos_in_browsed_path:
                self.detected_format_label.setText(f"未识别格式 (包含视频)")
                self.detected_format_label.setStyleSheet(
                    "font-weight: bold; color: #f39c12;")
                self.annotate_btn.setEnabled(True)
            elif detected == "Unknown":
                self.detected_format_label.setText(f"{detected}")
                self.detected_format_label.setStyleSheet("font-weight: bold; color: #e85555;")
            else:
                self.detected_format_label.setText(f"{detected}")
                self.detected_format_label.setStyleSheet("font-weight: bold; color: #55e88d;")
                self.analyze_btn.setEnabled(True)
                self.annotate_btn.setEnabled(True)

            self.populate_file_list()

    def on_split_finished(self, message):
        QMessageBox.information(self, "分割任务状态", message)
        self.set_ui_enabled(True)
        if self.split_thread:
            self.split_thread.quit()
            self.split_thread.wait()

    def populate_file_list(self, indices_to_show=None):
        self.file_list_widget.clear()
        self.all_image_annotations.clear()

        dataset_path = self.path_edit.text()
        if not dataset_path or not os.path.isdir(dataset_path):
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}

        detected_format_text = self.detected_format_label.text()
        detected_format = detected_format_text.split('(')[0].strip()

        if detected_format != "New Dataset" and detected_format != "Unknown" and "未识别格式" not in detected_format:
            if detected_format in self.readers_map:
                try:
                    ReaderClass = self.readers_map[detected_format]
                    reader = ReaderClass(dataset_path)
                    self.all_image_annotations, _ = reader.read()
                except Exception as e:
                    print(f"Warning: Could not read existing dataset annotations: {e}")
            else:
                print(f"Warning: No reader found for detected format: {detected_format}")

        scan_dirs = [dataset_path]
        for split in ['train', 'val', 'test']:
            split_image_dir = os.path.join(dataset_path, split, 'images')
            if os.path.isdir(split_image_dir):
                scan_dirs.append(split_image_dir)
            if os.path.isdir(os.path.join(dataset_path, split)) and \
               any(os.path.splitext(f)[1].lower() in image_extensions for f in os.listdir(os.path.join(dataset_path, split))):
                scan_dirs.append(os.path.join(dataset_path, split))


        unique_paths_in_list = set()

        for ann_obj in self.all_image_annotations:
            if ann_obj.image_path not in unique_paths_in_list:
                item = QListWidgetItem(QIcon(":/icons/image_icon.png"),
                                       os.path.basename(ann_obj.image_path))
                item.setData(Qt.UserRole, ann_obj)
                item.setData(Qt.WhatsThisRole, "image")
                self.file_list_widget.addItem(item)
                unique_paths_in_list.add(ann_obj.image_path)

        for current_scan_dir in set(scan_dirs):
            for filename in os.listdir(current_scan_dir):
                file_path = os.path.join(current_scan_dir, filename)
                if not os.path.isfile(file_path):
                    continue

                base_name, ext = os.path.splitext(filename)
                ext = ext.lower()

                if file_path in unique_paths_in_list:
                    continue

                if ext in image_extensions:
                    if not any(ann.image_path == file_path for ann in self.all_image_annotations):
                        try:
                            with Image.open(file_path) as img:
                                width, height = img.size
                            ann = ImageAnnotation(image_path=file_path, width=width, height=height, annotations=[])
                            self.all_image_annotations.append(ann)
                            item = QListWidgetItem(QIcon(":/icons/image_icon.png"), os.path.basename(file_path))
                            item.setData(Qt.UserRole, ann)
                            item.setData(Qt.WhatsThisRole, "image")
                            self.file_list_widget.addItem(item)
                            unique_paths_in_list.add(file_path)
                        except Exception as e:
                            print(f"Error reading image dimensions from {file_path}: {e}")

                elif ext in video_extensions:
                    if file_path not in self.all_video_paths:
                        self.all_video_paths.append(file_path)

                    item = QListWidgetItem(QIcon(":/icons/video_icon.png"), os.path.basename(file_path))
                    item.setData(Qt.UserRole, file_path)
                    item.setData(Qt.WhatsThisRole, "video")
                    self.file_list_widget.addItem(item)
                    unique_paths_in_list.add(file_path)

        if indices_to_show is not None:
            filtered_data = [self.all_image_annotations[i] for i in indices_to_show if i < len(self.all_image_annotations)]
            self.file_list_widget.clear()
            for ann_obj in filtered_data:
                item = QListWidgetItem(QIcon(":/icons/image_icon.png"), os.path.basename(ann_obj.image_path))
                item.setData(Qt.UserRole, ann_obj)
                item.setData(Qt.WhatsThisRole, "image")
                self.file_list_widget.addItem(item)


        self.annotate_btn.setEnabled(bool(self.all_image_annotations) or bool(self.all_video_paths))

    def open_split_tool(self):
        dialog = SplitDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.config:
            self.set_ui_enabled(False)
            self.log_output.append("开始数据集分割...")
            self.split_worker = SplitWorker(dialog.config)
            self.split_thread = QThread()
            self.split_worker.moveToThread(self.split_thread)
            self.split_thread.started.connect(self.split_worker.run)
            self.split_worker.finished.connect(self.on_split_finished)
            self.split_worker.progress.connect(self.progress_bar.setValue)
            self.split_worker.log.connect(self.log_output.append)
            self.split_thread.start()

    def on_file_double_clicked(self, index):
        item = self.file_list_widget.item(index.row())
        if not item:
            return

        file_type = item.data(Qt.WhatsThisRole)

        if file_type == "image":
            ann = item.data(Qt.UserRole)
            if ann and hasattr(ann, 'image_path') and os.path.isfile(ann.image_path):
                try:
                    if sys.platform == 'win32':
                        os.startfile(os.path.abspath(ann.image_path))
                    elif sys.platform == 'darwin':
                        subprocess.call(['open', os.path.abspath(ann.image_path)])
                    else:
                        subprocess.call(['xdg-open', os.path.abspath(ann.image_path)])
                except Exception as e:
                    QMessageBox.warning(self, "打开文件失败", f"无法打开图片: {e}")
        elif file_type == "video":
            video_path = item.data(Qt.UserRole)
            if video_path and os.path.isfile(video_path):
                config = self.gather_config(for_analysis=True)
                if not config: return

                detected_label = self.detected_format_label.text()
                if "未识别格式 (包含视频)" in detected_label:
                    config['is_new_dataset'] = True
                    config['input_format'] = 'YOLO'

                editor_window = AnnotationWindow(config, self.readers_map, self.writers_map,
                                                 self, video_to_import_path=video_path, initial_data=self.all_image_annotations)
                editor_window.exec()

                reply = QMessageBox.question(self, "刷新数据?",
                                             "您可能已经修改了标注，是否要重新分析数据集以更新统计信息？",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes: self.start_analysis()
                self.populate_file_list()

    def open_annotation_window(self):
        config = self.gather_config(for_analysis=True)
        if not config: return

        video_to_import = None
        if self.all_video_paths:
            video_basenames = [os.path.basename(p) for p in self.all_video_paths]
            dialog_options = ["否，只打开标注器"] + video_basenames
            video_chosen_text, ok = QInputDialog.getItem(
                self, "导入视频", "检测到当前目录下有视频文件，您想立即导入并标注其中一个吗？",
                dialog_options, 0, False
            )
            if ok and video_chosen_text != "否，只打开标注器":
                video_to_import_idx = video_basenames.index(video_chosen_text)
                video_to_import = self.all_video_paths[video_to_import_idx]
            elif not ok or video_chosen_text == "否，只打开标注器":
                video_to_import = None

        detected_label = self.detected_format_label.text()
        if "未识别格式 (包含视频)" in detected_label:
            config['is_new_dataset'] = True
            config['input_format'] = 'YOLO'

        editor_window = AnnotationWindow(config, self.readers_map, self.writers_map,
                                         self, video_to_import_path=video_to_import,
                                         initial_data=self.all_image_annotations)
        editor_window.exec()

        reply = QMessageBox.question(self, "刷新数据?", "您可能已经修改了标注，是否要重新分析数据集以更新统计信息？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes: self.start_analysis()
        self.populate_file_list()

    def save_preset(self):
        preset_name, ok = QInputDialog.getText(self, "保存预设", "请输入预设名称:")
        if ok and preset_name:
            preset_data = {}
            for aug_name, ctrl in self.controls.items():
                params_values = {}
                for param_name, widget in ctrl['params'].items():
                    if isinstance(widget, QSlider):
                        params_values[param_name] = widget.value()
                    elif isinstance(widget, QSpinBox):
                        params_values[param_name] = widget.value()
                    elif isinstance(widget, QLabel):
                        params_values[param_name] = int(widget.text())
                preset_data[aug_name] = {"enabled": ctrl['checkbox'].isChecked(), "params": params_values}
            filepath = os.path.join(self.presets_dir, f"{preset_name}.json")
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(preset_data, f, indent=4)
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已成功保存！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存预设失败: {e}")

    def load_preset(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "加载预设", self.presets_dir, "JSON 文件 (*.json)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                for aug_name, ctrl in self.controls.items():
                    if aug_name in loaded_data:
                        preset_for_aug = loaded_data[aug_name]
                        ctrl['checkbox'].setChecked(preset_for_aug.get("enabled", False))
                        for param_name, widget in ctrl['params'].items():
                            if param_name in preset_for_aug.get("params", {}):
                                if isinstance(widget, (QSlider, QSpinBox)):
                                    widget.setValue(preset_for_aug["params"][param_name])
                                elif isinstance(widget, QLabel):
                                    widget.setText(str(preset_for_aug["params"][param_name]))
                    else:
                        ctrl['checkbox'].setChecked(False)
                QMessageBox.information(self, "成功", f"预设 '{os.path.basename(filepath)}' 已成功加载！")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载预设失败: {e}")

    def start_analysis(self):
        config = self.gather_config(for_analysis=True)
        if not config: return
        self.set_ui_enabled(False)
        self.setCursor(QCursor(Qt.WaitCursor))
        self.log_output.clear()

        try:
            ReaderClass = self.readers_map[config['input_format']]
            reader = ReaderClass(config['dataset_path'])
            self.all_image_annotations, class_names = reader.read()
            self.populate_file_list()
        except Exception as e:
            QMessageBox.critical(self, "数据加载失败", f"无法读取数据集：{e}")
            self.set_ui_enabled(True)
            self.unsetCursor()
            return

        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(config, self.readers_map)
        self.analysis_thread._worker = self.analysis_worker
        self.analysis_worker.moveToThread(self.analysis_thread)
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_worker.log.connect(self.log_output.append)
        self.analysis_thread.start()

    def on_analysis_complete(self, results):
        self.set_ui_enabled(True)
        self.unsetCursor()
        if self.analysis_thread:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            self.analysis_thread = None
            self.analysis_worker = None
        if results:
            dashboard = DashboardDialog(results, self.all_image_annotations, self)
            dashboard.filter_requested.connect(self.populate_file_list)
            dashboard.exec()
        else:
            QMessageBox.critical(self, "分析失败", "分析过程中发生错误，请查看日志。")

    def start_augmentation(self):
        config = self.gather_config()
        if not config: return
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.thread = QThread()
        self.worker = AugmentationWorker(config, self.readers_map, self.writers_map)
        self.thread._worker = self.worker
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.log.connect(self.log_output.append)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.thread.start()

    def stop_augmentation(self):
        if self.worker: self.worker.stop()

    def on_task_finished(self, message):
        QMessageBox.information(self, "任务状态", message)
        self.set_ui_enabled(True)
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

    def set_ui_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.left_panel.setEnabled(enabled)
        self.right_panel.setEnabled(enabled)

    def closeEvent(self, event):
        running_threads = []
        threads_to_check = [self.thread, self.analysis_thread, self.split_thread]
        for t in threads_to_check:
            if t and t.isRunning():
                running_threads.append(t)
                if hasattr(t, '_worker') and hasattr(t._worker, 'stop'):
                    t._worker.stop()
                    self.log_output.append(f"正在停止线程 {t.__class__.__name__}...")
                t.quit()

        if running_threads:
            progress = QProgressDialog("正在等待后台任务结束...", "取消", 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("正在关闭")
            progress.show()

            for t in running_threads:
                if not t.wait(15000):
                    self.log_output.append(f"警告: 线程 {t.__class__.__name__} 未能在15秒内结束。可能需要手动终止。")
            progress.close()

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AugmentationApp()
    window.show()
    sys.exit(app.exec())