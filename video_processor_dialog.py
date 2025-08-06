import os
import time
import random
import cv2
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSpinBox, QFileDialog, QProgressBar, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QObject


# 确保您可以访问这些类
# from formats.internal_data import ImageAnnotation, Annotation, BBox # AnnotationWindow中已导入

class VideoProcessorWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, video_path, num_frames_to_extract, dataset_path):
        super().__init__()
        self.video_path = video_path
        self.num_frames_to_extract = num_frames_to_extract
        self.dataset_path = dataset_path
        self.is_cancelled = False

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频文件: {self.video_path}")
                self.finished.emit([])
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames == 0:
                self.error.emit("视频中没有可用的帧。")
                self.finished.emit([])
                return

            if self.num_frames_to_extract > total_frames:
                self.num_frames_to_extract = total_frames

            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
            frames_output_dir = os.path.join(self.dataset_path, "images", f"{video_basename}_frames_{int(time.time())}")
            os.makedirs(frames_output_dir, exist_ok=True)

            if self.num_frames_to_extract == total_frames:
                frame_indices_to_extract = list(range(total_frames))
            else:
                frame_indices_to_extract = sorted(random.sample(range(total_frames), self.num_frames_to_extract))

            extracted_annotations = []
            for i, frame_idx in enumerate(frame_indices_to_extract):
                if self.is_cancelled:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    self.progress.emit(i, f"警告: 无法读取帧 {frame_idx}")
                    continue

                frame_filename = f"{video_basename}_frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(frames_output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                from formats.internal_data import ImageAnnotation
                ann = ImageAnnotation(
                    image_path=frame_path,
                    width=width,
                    height=height,
                    annotations=[]
                )
                extracted_annotations.append(ann)
                self.progress.emit(i + 1,
                                   f"正在提取帧: {frame_idx}/{total_frames} ({i + 1}/{self.num_frames_to_extract} 已提取)")

            cap.release()
            self.progress.emit(self.num_frames_to_extract, "帧提取完成。")
            self.finished.emit(extracted_annotations)

        except Exception as e:
            self.error.emit(f"帧提取时发生错误: {e}")
            import traceback
            print(traceback.format_exc())
            self.finished.emit([])

    def cancel(self):
        self.is_cancelled = True


class VideoProcessorDialog(QDialog):
    def __init__(self, dataset_path, parent=None, initial_video_path=None):
        super().__init__(parent)
        self.setWindowTitle("从视频提取帧")
        self.setMinimumSize(400, 200)
        self.dataset_path = dataset_path
        self.extracted_annotations = []

        self.video_path_edit = QLineEdit()
        self.num_frames_spinbox = QSpinBox()
        self.num_frames_spinbox.setRange(1, 99999)
        self.num_frames_spinbox.setValue(100)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("等待开始...")

        self.browse_button = QPushButton("浏览...")
        self.extract_button = QPushButton("开始提取")
        self.cancel_button = QPushButton("取消")

        self.setup_ui()
        self.connect_signals()

        if initial_video_path:
            self.video_path_edit.setText(initial_video_path)
            self.update_spinbox_max_from_video(initial_video_path)

        self.worker_thread = None
        self.worker = None

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("视频文件:"))
        path_layout.addWidget(self.video_path_edit)
        path_layout.addWidget(self.browse_button)
        main_layout.addLayout(path_layout)

        frames_layout = QHBoxLayout()
        frames_layout.addWidget(QLabel("提取帧数:"))
        frames_layout.addWidget(self.num_frames_spinbox)
        main_layout.addLayout(frames_layout)

        main_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.extract_button)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

    def connect_signals(self):
        self.browse_button.clicked.connect(self.browse_video_file)
        self.extract_button.clicked.connect(self.start_extraction)
        self.cancel_button.clicked.connect(self.cancel_extraction)

    def update_spinbox_max_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.num_frames_spinbox.setMaximum(total_frames)
                self.num_frames_spinbox.setValue(min(self.num_frames_spinbox.value(), total_frames))
            else:
                QMessageBox.warning(self, "警告", "视频中没有可用的帧。")
                self.num_frames_spinbox.setMaximum(99999)
        else:
            QMessageBox.warning(self, "警告", "无法打开视频文件，请检查文件是否损坏或格式不支持。")
        cap.release()

    def browse_video_file(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
        )
        if video_path:
            self.video_path_edit.setText(video_path)
            self.update_spinbox_max_from_video(video_path)

    def start_extraction(self):
        video_path = self.video_path_edit.text()
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "错误", "请选择一个有效的视频文件。")
            return

        num_frames = self.num_frames_spinbox.value()

        self.set_ui_busy(True)
        self.progress_bar.setFormat("正在初始化...")
        self.progress_bar.setValue(0)

        self.worker_thread = QThread()
        self.worker = VideoProcessorWorker(video_path, num_frames, self.dataset_path)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_extraction_finished)
        self.worker.error.connect(self.on_extraction_error)

        self.worker_thread.start()

    def update_progress(self, current, message):
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(message)

    def on_extraction_finished(self, annotations):
        self.extracted_annotations = annotations
        self.progress_bar.setFormat("提取完成！")
        self.set_ui_busy(False)
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        QMessageBox.information(self, "完成", f"已成功从视频中提取 {len(annotations)} 帧。")
        self.accept()

    def on_extraction_error(self, message):
        QMessageBox.critical(self, "错误", message)
        self.set_ui_busy(False)
        self.progress_bar.setFormat("提取失败。")
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.reject()

    def cancel_extraction(self):
        if self.worker:
            self.worker.cancel()
        self.set_ui_busy(False)
        self.progress_bar.setFormat("已取消。")
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.reject()

    def set_ui_busy(self, busy):
        self.video_path_edit.setEnabled(not busy)
        self.num_frames_spinbox.setEnabled(not busy)
        self.browse_button.setEnabled(not busy)
        self.extract_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if busy:
            self.progress_bar.setMaximum(self.num_frames_spinbox.value())
            self.setCursor(Qt.WaitCursor)
        else:
            self.unsetCursor()
            self.progress_bar.setMaximum(100)

    def get_extracted_annotations(self):
        return self.extracted_annotations

    def closeEvent(self, event):
        self.cancel_extraction()
        super().closeEvent(event)