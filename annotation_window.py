import os
import cv2
import re
import numpy as np
import uuid
import logging
import traceback
import concurrent.futures
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QGraphicsView,
                               QGraphicsScene, QGraphicsPixmapItem, QPushButton, QInputDialog,
                               QMessageBox, QListWidgetItem, QMenu, QSplitter, QWidget, QFileDialog,
                               QToolBar, QMenuBar, QStatusBar, QComboBox,
                               QProgressDialog, QGraphicsTextItem, QLabel)
from PySide6.QtGui import QPixmap, QColor, QPainter, QCursor, QPen, QAction, QKeySequence, QTransform, \
    QActionGroup
from PySide6.QtCore import Qt, QRectF, QPointF, QSize, QObject, Signal, QThread
from PIL import Image

from class_manager_dialog import ClassManagerDialog
from formats.detector import detect_format
from formats.internal_data import ImageAnnotation, Annotation, BBox
from interactive_bbox_item import InteractiveBBoxItem
from ai_models import AI_AVAILABLE, run_sam_prediction, ModelManager
from video_processor_dialog import VideoProcessorDialog
from local_tracker_manager import LocalTrackerManager


class ImageLoaderWorker(QObject):
    finished = Signal(str, object)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            image_data = cv2.imread(self.image_path)
            if image_data is not None:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            self.finished.emit(self.image_path, image_data)
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")
            self.finished.emit(self.image_path, None)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]


class AnnotationView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.parent_window = parent
        self.show_crosshair = False
        self.crosshair_view_pos = None
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.Antialiasing)
        self.current_zoom_scale = 1.0
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def setCrosshairEnabled(self, enabled):
        self.show_crosshair = enabled
        if not enabled:
            self.crosshair_view_pos = None
        self.viewport().update()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.show_crosshair or self.crosshair_view_pos is None:
            return

        pen = QPen(QColor(255, 255, 0, 200), 1, Qt.DashLine)
        painter.setPen(pen)

        painter.save()
        painter.setTransform(QTransform())

        view_x = self.crosshair_view_pos.x()
        view_y = self.crosshair_view_pos.y()

        painter.drawLine(view_x, 0, view_x, self.viewport().height())
        painter.drawLine(0, view_y, self.viewport().width(), view_y)

        painter.restore()

    def mousePressEvent(self, event):
        if self.parent_window.handle_view_mouse_press(event):
            event.accept()
        else:
            super().mousePressEvent(event)
        self.viewport().update()

    def mouseMoveEvent(self, event):
        if self.show_crosshair:
            self.crosshair_view_pos = event.pos()
            self.viewport().update()

        if self.parent_window.handle_view_mouse_move(event):
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.parent_window.handle_view_mouse_release(event):
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self.viewport().update()

    def leaveEvent(self, event):
        self.crosshair_view_pos = None
        self.viewport().update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if event.modifiers() == Qt.ControlModifier:
            zoom_in_ratio = 1.25
            zoom_out_ratio = 0.8

            if event.angleDelta().y() > 0:
                new_scale = self.current_zoom_scale * zoom_in_ratio
            else:
                new_scale = self.current_zoom_scale * zoom_out_ratio

            new_scale = max(0.1, min(new_scale, 10.0))

            if abs(new_scale - self.current_zoom_scale) < 0.001:
                event.accept()
                return

            self.current_zoom_scale = new_scale
            current_actual_scale = self.transform().m11()
            relative_scale_factor = new_scale / current_actual_scale if current_actual_scale != 0 else new_scale
            self.scale(relative_scale_factor, relative_scale_factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def apply_stored_zoom(self):
        self.setTransform(QTransform().scale(self.current_zoom_scale, self.current_zoom_scale))


class AnnotationWindow(QDialog):
    def __init__(self, config, readers_map, writers_map, parent=None, video_to_import_path=None, initial_data=None):
        super().__init__(parent)
        self.setWindowTitle("数据集标注与编辑器")
        self.setMinimumSize(1600, 900)
        self.config = config
        self.readers_map = readers_map
        self.writers_map = writers_map
        self.all_data: list[ImageAnnotation] = initial_data if initial_data is not None else []
        self.class_names = []
        self.current_image_ann: ImageAnnotation = None
        self.class_colors = []
        self.sam_predictor = None
        self.current_pixmap_item = None
        self.mode = "edit"
        self.is_drawing = False
        self.current_class_id = 0
        self.temporary_sam_bbox_item = None

        self.tracker_manager = None
        self.selected_tracker_type = "CSRT"
        self.tracking_active = False
        self.current_image_index = -1

        self.image_loader_thread = None
        self.preloaded_image_data = None
        self.preloaded_image_path = None

        main_layout = QVBoxLayout(self)
        self.create_menus()
        self.create_toolbar()

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        self.class_list.currentRowChanged.connect(self.on_class_selected)
        left_layout.addWidget(QLabel("类别列表:"))
        left_layout.addWidget(self.class_list)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.list_widget.currentItemChanged.connect(self.display_image)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_image_list_context_menu)
        left_layout.addWidget(QLabel("图像列表:"))
        left_layout.addWidget(self.list_widget)

        splitter.addWidget(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪 | 模式: 手动标注 | 当前类别: 未选择")
        self.scene = QGraphicsScene()
        self.view = AnnotationView(self.scene, self)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        right_layout.addWidget(self.view, 1)
        right_layout.addWidget(self.status_bar)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)
        self.create_bottom_toolbar()

        if video_to_import_path:
            self.import_frames_from_video(video_to_import_path)
        else:
            if self.all_data:
                self._load_class_names_from_file(self.config['dataset_path'])
                if not self.class_names:
                    print("未找到 classes.txt 或文件为空，将根据标签文件自动生成类别名称。")
                    max_id = -1
                    for img_ann in self.all_data:
                        for ann in img_ann.annotations:
                            if ann.class_id > max_id:
                                max_id = ann.class_id
                    if max_id > -1:
                        self.class_names = [f'class_{i}' for i in range(max_id + 1)]
                    else:
                        print("警告：数据集中未找到任何有效标注，无法生成类别。")

                self.generate_class_colors()
                self.populate_image_list()
            else:
                if self.open_dataset():
                    self.update_ui_state()
                else:
                    QMessageBox.warning(self, "无数据", "未加载任何数据，请通过主窗口选择数据集。")

        self.update_ui_state()

    def _preload_next_image(self):
        if self.image_loader_thread and self.image_loader_thread.isRunning():
            return

        next_index = self.current_image_index + 1
        if 0 <= next_index < len(self.all_data):
            next_image_path = self.all_data[next_index].image_path
            if next_image_path == self.preloaded_image_path:
                return

            self.image_loader_thread = QThread()
            worker = ImageLoaderWorker(next_image_path)
            worker.moveToThread(self.image_loader_thread)
            worker.finished.connect(self._on_image_preloaded)
            self.image_loader_thread.started.connect(worker.run)
            self.image_loader_thread.start()

    def _on_image_preloaded(self, image_path, image_data):
        self.preloaded_image_path = image_path
        self.preloaded_image_data = image_data
        if self.image_loader_thread:
            self.image_loader_thread.quit()
            self.image_loader_thread.wait()
            self.image_loader_thread = None

    def show_image_list_context_menu(self, pos):
        pass

    def import_frames_from_video(self, video_path):
        processor_dialog = VideoProcessorDialog(self.config['dataset_path'], self, initial_video_path=video_path)
        if processor_dialog.exec() == QDialog.Accepted:
            extracted_anns = processor_dialog.get_extracted_annotations()
            if extracted_anns:
                self.all_data.extend(extracted_anns)
                self.populate_image_list()
                QMessageBox.information(self, "视频导入", f"已从视频导入 {len(extracted_anns)} 帧进行标注。")
            else:
                QMessageBox.warning(self, "视频导入", "未能从视频中提取任何帧。")
        else:
            QMessageBox.information(self, "视频导入", "视频帧提取已取消或失败。")
        self.update_ui_state()

    def open_dataset(self):
        dataset_path = QFileDialog.getExistingDirectory(self, "选择数据集目录", self.config.get('dataset_path', ''))
        if not dataset_path:
            return False
        return self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path=None):
        try:
            if dataset_path is None:
                if 'dataset_path' not in self.config:
                    return self.open_dataset()
                dataset_path = self.config.get('dataset_path')

            self.config['dataset_path'] = dataset_path
            self.all_data = []

            self._load_class_names_from_file(dataset_path)

            image_search_paths = [dataset_path, os.path.join(dataset_path, 'images')]
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            found_images = []
            for search_dir in image_search_paths:
                if not os.path.isdir(search_dir):
                    continue
                for filename in os.listdir(search_dir):
                    if os.path.splitext(filename)[1].lower() in image_extensions:
                        found_images.append(os.path.join(search_dir, filename))

            found_images.sort(key=natural_sort_key)

            for img_path in found_images:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                    ann = ImageAnnotation(image_path=img_path, width=width, height=height, annotations=[])
                    self._read_yolo_label_file(ann)
                    self.all_data.append(ann)
                except Exception as e:
                    print(f"Error processing image {os.path.basename(img_path)}: {e}")

            if not self.class_names and self.all_data:
                max_id = -1
                for img_ann in self.all_data:
                    for ann in img_ann.annotations:
                        if ann.class_id > max_id:
                            max_id = ann.class_id
                if max_id > -1:
                    self.class_names = [f'class_{i}' for i in range(max_id + 1)]

            self.generate_class_colors()
            self.populate_image_list()
            self.status_bar.showMessage(f"已加载数据集: {dataset_path}", 5000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载数据集时发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def _load_class_names_from_file(self, dataset_path):
        self.class_names = []
        txt_path = os.path.join(dataset_path, 'classes.txt')
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    self.class_names = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"无法读取 classes.txt: {e}")

    def _save_class_names_to_file(self):
        if 'dataset_path' not in self.config or not self.config['dataset_path']:
            QMessageBox.warning(self, "无法保存类别", "未设置数据集路径。")
            return

        txt_path = os.path.join(self.config['dataset_path'], 'classes.txt')
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                for name in self.class_names:
                    f.write(name + '\n')
        except Exception as e:
            QMessageBox.critical(self, "保存类别失败", f"无法将类别写入 'classes.txt':\n{e}")

    def _read_yolo_label_file(self, image_ann: ImageAnnotation):
        label_path = os.path.splitext(image_ann.image_path)[0] + ".txt"
        inferred_label_path = None
        base_dataset_path = self.config.get('dataset_path')
        if base_dataset_path and image_ann.image_path.startswith(base_dataset_path):
            relative_image_path = os.path.relpath(image_ann.image_path, base_dataset_path)
            parts = relative_image_path.split(os.sep)
            try:
                images_idx = parts.index('images')
                parts[images_idx] = 'labels'
                inferred_label_path = os.path.join(base_dataset_path, *parts)
                inferred_label_path = os.path.splitext(inferred_label_path)[0] + ".txt"
            except ValueError:
                pass

        final_label_path = label_path

        if inferred_label_path and os.path.exists(inferred_label_path):
            final_label_path = inferred_label_path
        elif os.path.exists(label_path):
            final_label_path = label_path
        else:
            return

        try:
            with open(final_label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # The check against len(self.class_names) will be done when displaying
                        # This allows loading annotations even before class names are finalized
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bbox = BBox(x_center, y_center, width, height)
                        annotation = Annotation(class_id=class_id, bbox=bbox)
                        image_ann.annotations.append(annotation)

        except Exception as e:
            print(f"读取标注文件出错 {os.path.basename(final_label_path)}: {e}")

    def generate_class_colors(self):
        if not self.class_names:
            self.class_colors = []
            return
        self.class_colors = [QColor.fromHsv((i * 359 // max(1, len(self.class_names))) % 359, 220, 255) for i in
                             range(len(self.class_names))]
        self.update_class_list()

    def populate_image_list(self):
        self.list_widget.clear()
        self.all_data.sort(key=lambda x: natural_sort_key(x.image_path))
        for ann_obj in self.all_data:
            item = QListWidgetItem(os.path.basename(ann_obj.image_path))
            item.setData(Qt.UserRole, ann_obj)
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
            self.current_image_index = 0

    def display_image(self, current_item, previous_item):
        if previous_item:
            previous_ann_obj = previous_item.data(Qt.UserRole)
            if previous_ann_obj:
                new_annotations = []
                for item in self.scene.items():
                    if isinstance(item, InteractiveBBoxItem):
                        scene_rect = item.mapToScene(item.rect()).boundingRect()
                        w = previous_ann_obj.width
                        h = previous_ann_obj.height
                        if w > 0 and h > 0 and scene_rect.width() > 0 and scene_rect.height() > 0:
                            x_center = np.clip((scene_rect.left() + scene_rect.width() / 2) / w, 0, 1)
                            y_center = np.clip((scene_rect.top() + scene_rect.height() / 2) / h, 0, 1)
                            width = np.clip(scene_rect.width() / w, 0, 1)
                            height = np.clip(scene_rect.height() / h, 0, 1)

                            min_px_dim = 5.0
                            min_width_norm = min_px_dim / w
                            min_height_norm = min_px_dim / h
                            width = max(width, min_width_norm)
                            height = max(height, min_height_norm)

                            x_center = np.clip(x_center, width / 2, 1 - width / 2)
                            y_center = np.clip(y_center, height / 2, 1 - height / 2)

                            new_bbox = BBox(x_center, y_center, width, height)
                            if item.annotation_data:
                                item.annotation_data.bbox = new_bbox
                                new_annotations.append(item.annotation_data)
                            else:
                                logging.warning("InteractiveBBoxItem 没有关联 annotation_data。")
                                new_annotations.append(Annotation(0, new_bbox))

                previous_ann_obj.annotations = new_annotations

        self.scene.clear()

        if not current_item:
            self.current_image_ann = None
            self.current_pixmap_item = None
            self.current_image_index = -1
            return

        self.current_image_ann = current_item.data(Qt.UserRole)
        pixmap = QPixmap(self.current_image_ann.image_path)
        self.current_pixmap_item = self.scene.addPixmap(pixmap)
        self.current_pixmap_item.setZValue(-1)
        self.view.setSceneRect(QRectF(pixmap.rect()))
        self.view.apply_stored_zoom()

        self.current_image_index = self.list_widget.currentRow()

        for ann in self.current_image_ann.annotations:
            self.add_bbox_to_scene(ann)

        self._preload_next_image()

    def add_bbox_to_scene(self, annotation: Annotation):
        if not self.class_names:
            return

        w = self.current_image_ann.width
        h = self.current_image_ann.height
        bbox = annotation.bbox
        xmin = (bbox.x_center - bbox.width / 2) * w
        ymin = (bbox.y_center - bbox.height / 2) * h
        rect_w = bbox.width * w
        rect_h = bbox.height * h
        rect = QRectF(xmin, ymin, rect_w, rect_h)
        class_id = annotation.class_id
        if not (0 <= class_id < len(self.class_names)):
            logging.warning(
                f"标注中的类别ID '{class_id}' 超出范围。已加载的类别数量为 {len(self.class_names)}。该标注将不会显示。")
            return
        class_name = self.class_names[class_id]
        class_color = self.class_colors[class_id]
        bbox_item = InteractiveBBoxItem(rect, annotation, class_name, class_color)
        self.scene.addItem(bbox_item)

    # (The rest of the file remains unchanged, copy from the previous response)

    def show_bbox_context_menu(self, global_pos: QPointF):
        selected_items = [i for i in self.scene.selectedItems() if isinstance(i, InteractiveBBoxItem)]
        if not selected_items:
            item_under_cursor = self.view.itemAt(self.view.mapFromGlobal(global_pos))
            if isinstance(item_under_cursor, (InteractiveBBoxItem, QGraphicsTextItem)):
                parent_item = item_under_cursor if isinstance(item_under_cursor,
                                                              InteractiveBBoxItem) else item_under_cursor.parentItem()
                if parent_item and isinstance(parent_item, InteractiveBBoxItem):
                    selected_items = [parent_item]
        if not selected_items:
            return
        menu = QMenu()
        delete_bbox_action = menu.addAction("删除此标注")
        action = menu.exec(global_pos)
        if action == delete_bbox_action:
            for sel_item in selected_items:
                self.scene.removeItem(sel_item)
                if sel_item.annotation_data in self.current_image_ann.annotations:
                    self.current_image_ann.annotations.remove(sel_item.annotation_data)

    def _update_current_image_model_from_scene(self):
        if not self.current_image_ann:
            return
        new_annotations = []
        for item in self.scene.items():
            if isinstance(item, InteractiveBBoxItem):
                scene_rect = item.mapToScene(item.rect()).boundingRect()
                w = self.current_image_ann.width
                h = self.current_image_ann.height
                if w > 0 and h > 0 and scene_rect.width() > 0 and scene_rect.height() > 0:
                    new_bbox = BBox(x_center=(scene_rect.left() + scene_rect.width() / 2) / w,
                                    y_center=(scene_rect.top() + scene_rect.height() / 2) / h,
                                    width=scene_rect.width() / w, height=scene_rect.height() / h)
                    item.annotation_data.bbox = new_bbox
                    new_annotations.append(item.annotation_data)
        self.current_image_ann.annotations = new_annotations

    def save_current_annotations(self):
        self._update_current_image_model_from_scene()
        if not self.current_image_ann:
            return
        try:
            self._write_yolo_label_file(self.current_image_ann)
            base_name = os.path.splitext(os.path.basename(self.current_image_ann.image_path))[0]
            self.status_bar.showMessage(f"对 {base_name} 的修改已保存！", 3000)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存修改时发生错误: {e}")

    def save_all_annotations(self):
        if self.current_image_ann:
            self._update_current_image_model_from_scene()

        if not self.all_data:
            QMessageBox.information(self, "提示", "数据集中没有需要保存的标注。")
            return

        progress = QProgressDialog("正在保存所有标注...", "取消", 0, len(self.all_data), self)
        progress.setWindowTitle("保存全部")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        saved_count = 0
        errors = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_ann = {executor.submit(self._write_yolo_label_file, image_ann): image_ann for image_ann in
                             self.all_data}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_ann)):
                if progress.wasCanceled():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                image_ann = future_to_ann[future]
                progress.setValue(i + 1)
                progress.setLabelText(f"正在保存: {os.path.basename(image_ann.image_path)}")

                try:
                    future.result()
                    saved_count += 1
                except Exception as e:
                    errors.append(f"{os.path.basename(image_ann.image_path)}: {e}")

        progress.close()

        if errors:
            error_details = "\n".join(errors[:10])
            if len(errors) > 10:
                error_details += "\n...等"
            QMessageBox.warning(self, "部分保存失败",
                                f"成功保存 {saved_count} 个文件，但有 {len(errors)} 个文件保存失败。\n\n部分错误详情:\n{error_details}")
        elif progress.wasCanceled():
            QMessageBox.information(self, "已取消", f"用户取消了操作。在取消前已保存 {saved_count} 个文件的标注。")
        else:
            QMessageBox.information(self, "完成", f"已成功保存全部 {saved_count} 个文件的标注。")

    def handle_new_bbox_drawn(self, rect: QRectF):
        if not self.class_names:
            QMessageBox.warning(self, "错误", "无法创建新标注，因为没有可用的类别。请先管理类别。")
            return
        if not (0 <= self.current_class_id < len(self.class_names)):
            self.current_class_id = 0
            self.status_bar.showMessage(f"已自动选择类别: {self.class_names[0]}", 3000)
        class_id = self.current_class_id
        w = self.current_image_ann.width
        h = self.current_image_ann.height

        xmin_norm = np.clip(rect.left() / w, 0, 1)
        ymin_norm = np.clip(rect.top() / h, 0, 1)
        xmax_norm = np.clip(rect.right() / w, 0, 1)
        ymax_norm = np.clip(rect.bottom() / h, 0, 1)

        width_norm = xmax_norm - xmin_norm
        height_norm = ymax_norm - ymin_norm
        x_center_norm = xmin_norm + width_norm / 2
        y_center_norm = ymin_norm + height_norm / 2

        min_px_dim = 5.0
        if width_norm * w < min_px_dim: width_norm = min_px_dim / w
        if height_norm * h < min_px_dim: height_norm = min_px_dim / h

        x_center_norm = np.clip(x_center_norm, width_norm / 2, 1 - width_norm / 2)
        y_center_norm = np.clip(y_center_norm, height_norm / 2, 1 - height_norm / 2)

        bbox = BBox(x_center_norm, y_center_norm, width_norm, height_norm)

        new_track_id = str(uuid.uuid4().hex)[:8]
        track_name = f"新对象_{new_track_id}"

        for item in self.scene.items():
            if isinstance(item, InteractiveBBoxItem):
                item.setSelected(False)

        new_ann = Annotation(class_id, bbox, track_id=new_track_id, track_name=track_name)
        self.current_image_ann.annotations.append(new_ann)
        self.add_bbox_to_scene(new_ann)
        for item in self.scene.items():
            if isinstance(item, InteractiveBBoxItem) and item.annotation_data == new_ann:
                item.setSelected(True)
                item.setFocus()
                break

    def set_mode(self, new_mode):
        self.mode = new_mode
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setCrosshairEnabled(False)
        self.view.setCursor(Qt.ArrowCursor)

        if new_mode == "edit":
            self.view.setCursor(Qt.CrossCursor)
            self.view.setDragMode(QGraphicsView.RubberBandDrag)
            self.view.setCrosshairEnabled(True)
        elif new_mode == "sam":
            self.view.setCursor(Qt.PointingHandCursor)
            if self.sam_predictor is None and AI_AVAILABLE:
                self.setCursor(QCursor(Qt.WaitCursor))
                try:
                    self.sam_predictor = ModelManager(model_name='sam_vit_h', model_type='sam')
                    if self.sam_predictor is None:
                        QMessageBox.critical(self, "SAM模型加载失败", "请确保SAM模型权重文件 'sam_vit_h.pth' 已下载。")
                        self.set_mode("edit")
                except Exception as e:
                    QMessageBox.critical(self, "SAM模型加载错误", f"加载SAM模型时出错: {str(e)}")
                    self.set_mode("edit")
                finally:
                    self.unsetCursor()
        elif new_mode == "tracking":
            self.view.setCursor(Qt.ArrowCursor)
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.setCrosshairEnabled(False)

        self.update_ui_state()

    def keyPressEvent(self, event):
        key = event.key()
        if key in [Qt.Key_Delete, Qt.Key_Backspace]:
            self.delete_selected_annotations()
        elif key == Qt.Key_D:
            if self.list_widget.count() > 0:
                self.navigate_image(1, force_track=self.tracking_active)
        elif key == Qt.Key_A:
            if self.list_widget.count() > 0:
                self.navigate_image(-1, force_track=False)
        elif key == Qt.Key_Q:
            self.set_mode("edit")
        elif key == Qt.Key_E and AI_AVAILABLE:
            self.set_mode("sam")
        elif key == Qt.Key_W and self.mode == "edit":
            if not self.current_image_ann or not self.class_names:
                return
            w = self.current_image_ann.width
            h = self.current_image_ann.height
            rect = QRectF(w / 4, h / 4, w / 2, h / 2)
            self.handle_new_bbox_drawn(rect)
        elif Qt.Key_0 <= key <= Qt.Key_9:
            class_id = key - Qt.Key_0
            if class_id < len(self.class_names):
                self.current_class_id = class_id
                self.class_list.setCurrentRow(class_id)
                self.class_combo.setCurrentIndex(class_id)
                self.status_bar.showMessage(f"当前类别已切换为: {self.class_names[class_id]}", 3000)
                for item in self.scene.selectedItems():
                    if isinstance(item, InteractiveBBoxItem):
                        item.annotation_data.class_id = class_id
                        item.class_name = self.class_names[class_id]
                        item.text_item.setPlainText(item._get_display_text())
                        item.update()
        else:
            super().keyPressEvent(event)

    def navigate_image(self, direction: int, force_track: bool = False):
        if self.list_widget.count() == 0:
            return

        current_row = self.list_widget.currentRow()
        new_row = current_row + direction

        if 0 <= new_row < self.list_widget.count():
            self.list_widget.setCurrentRow(new_row)
            if self.tracking_active and force_track:
                self.run_tracking_on_current_frame()
        else:
            self.status_bar.showMessage("已到达图像列表末尾或开头。", 2000)
            if self.tracking_active:
                QMessageBox.information(self, "跟踪结束", "已到达视频/序列的末尾，跟踪已停止。")
                self.stop_tracking_mode()

    def run_tracking_on_current_frame(self):
        if not self.tracking_active or not self.tracker_manager or not self.current_image_ann:
            return

        image_data_rgb = None
        if self.preloaded_image_path == self.current_image_ann.image_path and self.preloaded_image_data is not None:
            image_data_rgb = self.preloaded_image_data
            self.preloaded_image_path = None
            self.preloaded_image_data = None
        else:
            image_data_bgr = cv2.imread(self.current_image_ann.image_path)
            if image_data_bgr is None:
                QMessageBox.critical(self, "错误", f"无法加载图像进行跟踪更新: {self.current_image_ann.image_path}")
                self.stop_tracking_mode()
                return
            image_data_rgb = cv2.cvtColor(image_data_bgr, cv2.COLOR_BGR2RGB)

        self.setCursor(QCursor(Qt.WaitCursor))
        try:
            items_to_remove = [item for item in self.scene.items() if
                               isinstance(item, InteractiveBBoxItem) or isinstance(item, QGraphicsTextItem)]
            for item in items_to_remove:
                self.scene.removeItem(item)

            predicted_annotations = self.tracker_manager.update_trackers(image_data_rgb)

            self.current_image_ann.annotations = predicted_annotations
            for ann in predicted_annotations:
                self.add_bbox_to_scene(ann)

            current_annotations_in_scene = [item.annotation_data for item in self.scene.items() if
                                            isinstance(item, InteractiveBBoxItem)]
            self.tracker_manager.init_trackers(image_data_rgb, current_annotations_in_scene)

            self.unsetCursor()
            self.status_bar.showMessage(f"已更新至下一帧并跟踪: {os.path.basename(self.current_image_ann.image_path)}",
                                        3000)
        except Exception as e:
            self.unsetCursor()
            QMessageBox.critical(self, "跟踪失败", f"跟踪下一帧时发生错误: {str(e)}\n{traceback.format_exc()}")
            self.stop_tracking_mode()

    def start_tracking_mode(self):
        if not self.current_image_ann:
            QMessageBox.warning(self, "错误", "请先加载图像。")
            return

        selected_bbox_items = [item for item in self.scene.selectedItems() if isinstance(item, InteractiveBBoxItem)]

        if not selected_bbox_items:
            QMessageBox.warning(self, "开始跟踪",
                                "请选择至少一个边界框以开始跟踪。请在编辑模式下，点击并拖动以绘制边界框，或者点击现有边界框以选中。")
            logging.warning("尝试开始跟踪，但未选中任何 InteractiveBBoxItem。")
            return

        annotations_to_track = []
        min_px_dim = 5.0

        for item in selected_bbox_items:
            scene_rect = item.mapToScene(item.rect()).boundingRect()
            w_img = self.current_image_ann.width
            h_img = self.current_image_ann.height

            if w_img <= 0 or h_img <= 0:
                logging.warning(
                    f"图像尺寸无效 (W:{w_img}, H:{h_img})，跳过边界框 {item.annotation_data.track_id} 的初始化。")
                continue

            if scene_rect.width() < min_px_dim or scene_rect.height() < min_px_dim:
                logging.warning(
                    f"场景中边界框 {item.annotation_data.track_id} 的尺寸过小 (W:{scene_rect.width()}, H:{scene_rect.height()})，跳过初始化。")
                continue

            normalized_x_center = np.clip((scene_rect.left() + scene_rect.width() / 2) / w_img, 0, 1)
            normalized_y_center = np.clip((scene_rect.top() + scene_rect.height() / 2) / h_img, 0, 1)
            normalized_width = np.clip(scene_rect.width() / w_img, 0, 1)
            normalized_height = np.clip(scene_rect.height() / h_img, 0, 1)

            min_width_norm = min_px_dim / w_img
            min_height_norm = min_px_dim / h_img
            normalized_width = max(normalized_width, min_width_norm)
            normalized_height = max(normalized_height, min_height_norm)

            normalized_x_center = np.clip(normalized_x_center, normalized_width / 2, 1 - normalized_width / 2)
            normalized_y_center = np.clip(normalized_y_center, normalized_height / 2, 1 - normalized_height / 2)

            normalized_bbox = BBox(normalized_x_center, normalized_y_center, normalized_width, normalized_height)

            track_id = item.annotation_data.track_id if item.annotation_data.track_id is not None else str(
                uuid.uuid4().hex)[:8]
            track_name = item.annotation_data.track_name if item.annotation_data.track_name else f"对象_{track_id}"

            temp_ann = Annotation(
                class_id=item.annotation_data.class_id,
                bbox=normalized_bbox,
                track_id=track_id,
                track_name=track_name
            )
            annotations_to_track.append(temp_ann)

        logging.info(f"准备初始化跟踪器，共找到 {len(annotations_to_track)} 个有效标注。")

        if not annotations_to_track:
            QMessageBox.warning(self, "开始跟踪",
                                "未能从选中的边界框中获取有效标注。请确保所选框在图像范围内且尺寸足够大。")
            return

        image_data_bgr = cv2.imread(self.current_image_ann.image_path)
        if image_data_bgr is None:
            QMessageBox.critical(self, "错误", f"无法加载图像进行跟踪: {self.current_image_ann.image_path}")
            return

        image_data_rgb = cv2.cvtColor(image_data_bgr, cv2.COLOR_BGR2RGB)

        self.setCursor(QCursor(Qt.WaitCursor))
        try:
            self.tracker_manager = LocalTrackerManager(tracker_type=self.selected_tracker_type)
            self.tracker_manager.init_trackers(image_data_rgb, annotations_to_track)

            for item in self.scene.items():
                if isinstance(item, InteractiveBBoxItem):
                    item.setSelected(False)

            self.tracking_active = True
            self.set_mode("tracking")
            self.status_bar.showMessage(f"跟踪模式已激活，使用 {self.selected_tracker_type} 跟踪。", 5000)
            self.list_widget.setFocus()
        except Exception as e:
            QMessageBox.critical(self, "跟踪器错误", f"初始化跟踪器时发生错误: {str(e)}\n{traceback.format_exc()}")
            self.stop_tracking_mode()
        finally:
            self.unsetCursor()

    def stop_tracking_mode(self):
        self.tracking_active = False
        if self.tracker_manager:
            self.tracker_manager.reset()
            self.tracker_manager = None
        self.set_mode("edit")
        self.status_bar.showMessage("跟踪模式已停用。", 3000)

    def handle_view_mouse_press(self, event):
        item_under_mouse = self.view.itemAt(event.pos())
        is_on_existing_item = isinstance(item_under_mouse, InteractiveBBoxItem) or (
                item_under_mouse and isinstance(item_under_mouse.parentItem(), InteractiveBBoxItem))

        if self.mode == 'tracking':
            if is_on_existing_item:
                return False
            elif event.button() == Qt.LeftButton:
                self.view.setDragMode(QGraphicsView.ScrollHandDrag)
                super().mousePressEvent(event)
                return True
            return False
        elif self.mode == 'edit' and is_on_existing_item:
            return False
        elif self.mode == 'sam' and event.button() == Qt.LeftButton:
            if self.temporary_sam_bbox_item:
                self.scene.removeItem(self.temporary_sam_bbox_item)
                self.temporary_sam_bbox_item = None
            if not self.sam_predictor or not self.class_names:
                QMessageBox.warning(self, "错误", "SAM或类别未初始化。")
                self.set_mode("edit")
                return False
            self.setCursor(QCursor(Qt.WaitCursor))
            try:
                point = self.view.mapToScene(event.pos())
                image_data = cv2.imread(self.current_image_ann.image_path)
                if image_data is None:
                    QMessageBox.warning(self, "错误", f"无法加载图片: {self.current_image_ann.image_path}")
                    return False
                point_coords = np.array([[point.x(), point.y()]])
                bbox_px = run_sam_prediction(self.sam_predictor, image_data, point_coords)
                if bbox_px:
                    temp_rect = QRectF(bbox_px[0], bbox_px[1], bbox_px[2] - bbox_px[0], bbox_px[3] - bbox_px[1])
                    self.handle_new_bbox_drawn(temp_rect)
                    return True
                else:
                    QMessageBox.information(self, "SAM", "SAM未能找到有效的分割。请尝试在不同位置点击。")
                    return True
            except Exception as e:
                QMessageBox.critical(self, "SAM预测错误", f"运行SAM预测时出错: {str(e)}")
                return False
            finally:
                self.unsetCursor()
        elif self.mode == 'edit' and event.button() == Qt.LeftButton:
            if not self.class_names:
                QMessageBox.warning(self, "错误", "无法创建新标注，因为没有可用的类别。")
                return False
            self.is_drawing = True
            self.view.start_point = self.view.mapToScene(event.pos())
            rect = QRectF(self.view.start_point, self.view.start_point)
            color = self.class_colors[self.current_class_id] if 0 <= self.current_class_id < len(
                self.class_colors) else Qt.yellow
            pen = QPen(color, 2, Qt.DashLine)
            self.view.current_rect_item = self.scene.addRect(rect, pen)
            return True
        return False

    def handle_view_mouse_move(self, event):
        if self.mode == 'tracking':
            self.view.viewport().update()
            return False
        elif self.is_drawing and self.mode == 'edit':
            end_point = self.view.mapToScene(event.pos())
            rect = QRectF(self.view.start_point, end_point).normalized()
            self.view.current_rect_item.setRect(rect)
            return True
        return False

    def handle_view_mouse_release(self, event):
        if self.mode == 'tracking':
            return False
        elif self.is_drawing and self.mode == 'edit' and event.button() == Qt.LeftButton:
            self.is_drawing = False
            if hasattr(self.view, 'current_rect_item') and self.view.current_rect_item:
                final_rect = self.view.current_rect_item.rect()
                self.scene.removeItem(self.view.current_rect_item)
                self.view.current_rect_item = None
                if final_rect.width() > 1 and final_rect.height() > 1:
                    self.handle_new_bbox_drawn(final_rect)
            return True
        return False

    def create_menus(self):
        menubar = QMenuBar(self)
        self.layout().setMenuBar(menubar)
        file_menu = menubar.addMenu("文件(&F)")
        open_action = QAction("打开数据集...", self)
        open_action.triggered.connect(self.open_dataset)
        file_menu.addAction(open_action)
        save_action = QAction("保存(&S)", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_current_annotations)
        file_menu.addAction(save_action)
        save_all_action = QAction("保存全部", self)
        save_all_action.triggered.connect(self.save_all_annotations)
        file_menu.addAction(save_all_action)
        file_menu.addSeparator()
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        edit_menu = menubar.addMenu("编辑(&E)")
        manage_classes_action = QAction("管理类别...", self)
        manage_classes_action.triggered.connect(self.manage_classes)
        edit_menu.addAction(manage_classes_action)
        view_menu = menubar.addMenu("视图(&V)")
        mode_menu = view_menu.addMenu("模式")
        self.edit_mode_action = QAction("手动标注 (Q)", self, checkable=True)
        self.edit_mode_action.setShortcut("Q")
        self.edit_mode_action.triggered.connect(lambda: self.set_mode("edit"))
        mode_menu.addAction(self.edit_mode_action)
        self.sam_mode_action = QAction("SAM自动标注 (E)", self, checkable=True)
        self.sam_mode_action.setShortcut("E")
        self.sam_mode_action.triggered.connect(lambda: self.set_mode("sam"))
        mode_menu.addAction(self.sam_mode_action)
        self.tracking_mode_action = QAction("跟踪模式", self, checkable=True)
        self.tracking_mode_action.triggered.connect(lambda: self.set_mode("tracking"))
        mode_menu.addAction(self.tracking_mode_action)
        self.mode_group = QActionGroup(self)
        self.mode_group.addAction(self.edit_mode_action)
        self.mode_group.addAction(self.sam_mode_action)
        self.mode_group.addAction(self.tracking_mode_action)
        self.mode_group.setExclusive(True)
        view_menu.addSeparator()
        self.toolbar_action = QAction("显示工具栏", self, checkable=True)
        self.toolbar_action.setChecked(True)
        self.toolbar_action.triggered.connect(self.toggle_toolbar)
        view_menu.addAction(self.toolbar_action)
        tracking_menu = menubar.addMenu("跟踪(&T)")
        tracker_type_menu = tracking_menu.addMenu("选择跟踪器")
        self.tracker_type_group = QActionGroup(self)
        self.tracker_type_actions = {}
        for tracker_name in LocalTrackerManager.TRACKER_TYPES.keys():
            action = QAction(tracker_name, self, checkable=True)
            action.triggered.connect(lambda checked, name=tracker_name: self.set_tracker_type(name))
            self.tracker_type_group.addAction(action)
            tracker_type_menu.addAction(action)
            self.tracker_type_actions[tracker_name] = action
        self.tracker_type_actions[self.selected_tracker_type].setChecked(True)
        tracking_menu.addSeparator()
        self.start_tracking_action = QAction("开始跟踪", self)
        self.start_tracking_action.triggered.connect(self.start_tracking_mode)
        tracking_menu.addAction(self.start_tracking_action)
        self.stop_tracking_action = QAction("停止跟踪", self)
        self.stop_tracking_action.triggered.connect(self.stop_tracking_mode)
        tracking_menu.addAction(self.stop_tracking_action)

    def create_toolbar(self):
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setIconSize(QSize(24, 24))
        self.layout().insertWidget(0, self.toolbar)
        self.toolbar.addAction(self.edit_mode_action)
        self.toolbar.addAction(self.sam_mode_action)
        self.toolbar.addAction(self.tracking_mode_action)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel("跟踪器:"))
        self.tracker_type_combo = QComboBox()
        for name in LocalTrackerManager.TRACKER_TYPES.keys():
            self.tracker_type_combo.addItem(name)
        self.tracker_type_combo.setCurrentText(self.selected_tracker_type)
        self.tracker_type_combo.currentTextChanged.connect(self.set_tracker_type)
        self.toolbar.addWidget(self.tracker_type_combo)
        self.toolbar.addAction(self.start_tracking_action)
        self.toolbar.addAction(self.stop_tracking_action)
        self.toolbar.addSeparator()
        manage_classes_btn = QAction("管理类别", self)
        manage_classes_btn.triggered.connect(self.manage_classes)
        self.toolbar.addAction(manage_classes_btn)
        self.toolbar.addSeparator()
        save_btn = QAction("保存", self)
        save_btn.setShortcut(QKeySequence.Save)
        save_btn.triggered.connect(self.save_current_annotations)
        self.toolbar.addAction(save_btn)
        save_all_btn = QAction("保存全部", self)
        save_all_btn.triggered.connect(self.save_all_annotations)
        self.toolbar.addAction(save_all_btn)

    def create_bottom_toolbar(self):
        bottom_toolbar = QToolBar("底部工具栏")
        self.layout().addWidget(bottom_toolbar)
        prev_btn = QAction("上一张 (A)", self)
        prev_btn.triggered.connect(lambda: self.navigate_image(-1))
        bottom_toolbar.addAction(prev_btn)
        next_btn = QAction("下一张 (D)", self)
        next_btn.triggered.connect(lambda: self.navigate_image(1))
        bottom_toolbar.addAction(next_btn)
        self.track_next_frame_action = QAction("下一帧跟踪 (D)", self)
        self.track_next_frame_action.triggered.connect(lambda: self.navigate_image(1, force_track=True))
        self.track_next_frame_action.setEnabled(False)
        bottom_toolbar.addAction(self.track_next_frame_action)
        bottom_toolbar.addSeparator()
        delete_btn = QAction("删除选中标注 (Del)", self)
        delete_btn.triggered.connect(self.delete_selected_annotations)
        bottom_toolbar.addAction(delete_btn)
        bottom_toolbar.addSeparator()
        bottom_toolbar.addWidget(QLabel("  当前类别: "))
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.on_class_selected_from_combo)
        bottom_toolbar.addWidget(self.class_combo)
        self.update_class_combo()

    def update_class_list(self):
        self.class_list.clear()
        self.class_list.addItems(self.class_names)
        if 0 <= self.current_class_id < len(self.class_names):
            self.class_list.setCurrentRow(self.current_class_id)
        self.update_class_combo()

    def update_class_combo(self):
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        if 0 <= self.current_class_id < len(self.class_names):
            self.class_combo.setCurrentIndex(self.current_class_id)

    def on_class_selected(self, index):
        if 0 <= index < len(self.class_names):
            self.current_class_id = index
            self.class_combo.setCurrentIndex(index)
            self.update_ui_state()

    def on_class_selected_from_combo(self, index):
        if 0 <= index < len(self.class_names):
            self.current_class_id = index
            self.class_list.setCurrentRow(index)
            self.update_ui_state()
            for item in self.scene.selectedItems():
                if isinstance(item, InteractiveBBoxItem):
                    item.annotation_data.class_id = index
                    item.class_name = self.class_names[index]
                    item.text_item.setPlainText(item._get_display_text())
                    item.update()

    def manage_classes(self):
        if not hasattr(self, 'class_names') or self.class_names is None:
            self.class_names = []
        dialog = ClassManagerDialog(self.class_names, self)
        if dialog.exec() == QDialog.Accepted:
            self.class_names = dialog.class_names
            self.generate_class_colors()
            self.update_class_list()
            self._save_class_names_to_file()
            self.status_bar.showMessage("类别已更新并保存至 classes.txt", 3000)
            if self.current_image_ann:
                self.display_image(self.list_widget.currentItem(), None)

    def delete_selected_annotations(self):
        if not self.current_image_ann:
            return
        selected_items = [item for item in self.scene.selectedItems() if isinstance(item, InteractiveBBoxItem)]
        if not selected_items:
            return
        for item in selected_items:
            try:
                if item.annotation_data in self.current_image_ann.annotations:
                    self.current_image_ann.annotations.remove(item.annotation_data)
                self.scene.removeItem(item)
            except (ValueError, AttributeError):
                pass

    def toggle_toolbar(self):
        self.toolbar.setVisible(self.toolbar_action.isChecked())

    def set_tracker_type(self, tracker_name):
        self.selected_tracker_type = tracker_name
        for action_item in self.tracker_type_actions.values():
            action_item.setChecked(action_item.text() == tracker_name)
        self.tracker_type_combo.setCurrentText(tracker_name)
        self.status_bar.showMessage(f"已选择跟踪器: {tracker_name}", 3000)
        self.update_ui_state()

    def update_ui_state(self):
        self.edit_mode_action.setChecked(self.mode == "edit")
        self.sam_mode_action.setChecked(self.mode == "sam")
        self.tracking_mode_action.setChecked(self.mode == "tracking")

        is_image_loaded = bool(self.current_image_ann)

        self.start_tracking_action.setEnabled(is_image_loaded and not self.tracking_active)
        self.stop_tracking_action.setEnabled(self.tracking_active)
        self.track_next_frame_action.setEnabled(self.tracking_active)
        self.tracker_type_combo.setEnabled(not self.tracking_active)
        self.sam_mode_action.setEnabled(AI_AVAILABLE)

        if self.mode == "edit":
            self.view.setCursor(Qt.CrossCursor)
            self.view.setDragMode(QGraphicsView.RubberBandDrag)
            self.view.setCrosshairEnabled(True)
        elif self.mode == "sam":
            self.view.setCursor(Qt.PointingHandCursor)
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.setCrosshairEnabled(False)
        elif self.mode == "tracking":
            self.view.setCursor(Qt.ArrowCursor)
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.setCrosshairEnabled(False)
        else:
            self.view.setCursor(Qt.ArrowCursor)
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.setCrosshairEnabled(False)

        mode_text = ""
        if self.mode == "edit":
            mode_text = "手动标注"
        elif self.mode == "sam":
            mode_text = "SAM自动标注"
        elif self.mode == "tracking":
            mode_text = "跟踪模式"

        class_text = self.class_names[self.current_class_id] if self.class_names and 0 <= self.current_class_id < len(
            self.class_names) else "未选择"

        status_message = f"模式: {mode_text}"
        if self.mode == "tracking":
            status_message += f" | 跟踪器: {self.selected_tracker_type}"
        status_message += f" | 类别: {class_text}"
        self.status_bar.showMessage(status_message)

    def closeEvent(self, event):
        super().closeEvent(event)