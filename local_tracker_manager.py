import cv2
import numpy as np
import logging
import uuid
from formats.internal_data import ImageAnnotation, Annotation, BBox


class LocalTrackerManager:
    TRACKER_TYPES = {
        'CSRT': None,
        'KCF': None,
        'MIL': None,
        'Boosting': None,
        'MedianFlow': None,
        'MOSSE': None,
        'TLD': None,
    }

    def __init__(self, tracker_type: str = "CSRT"):
        # 尝试从 cv2.legacy 获取跟踪器，如果不存在则从 cv2 获取
        for name in self.TRACKER_TYPES.keys():
            try:
                self.TRACKER_TYPES[name] = getattr(cv2.legacy, f"Tracker{name}_create")
            except AttributeError:
                try:
                    self.TRACKER_TYPES[name] = getattr(cv2, f"Tracker{name}_create")
                except AttributeError:
                    logging.error(f"无法找到跟踪器 {name} 的创建函数。请确保已安装 opencv-contrib-python 并且版本兼容。")
                    self.TRACKER_TYPES[name] = None  # 标记为不可用

        if tracker_type not in self.TRACKER_TYPES or self.TRACKER_TYPES[tracker_type] is None:
            raise ValueError(f"不支持或无法加载跟踪器类型: {tracker_type}。请检查您的OpenCV安装。")

        self.tracker_creator = self.TRACKER_TYPES[tracker_type]
        self.trackers = []

    def init_trackers(self, frame_rgb: np.array, annotations: list):
        self.trackers = []

        if not annotations:
            logging.warning("没有提供标注用于初始化跟踪器。")
            return

        h, w, _ = frame_rgb.shape

        for ann in annotations:
            x_center_px = ann.bbox.x_center * w
            y_center_px = ann.bbox.y_center * h
            width_px = ann.bbox.width * w
            height_px = ann.bbox.height * h

            x = int(x_center_px - width_px / 2)
            y = int(y_center_px - height_px / 2)
            bbox_cv = (x, y, int(width_px), int(height_px))

            x_min = max(0, x)
            y_min = max(0, y)
            x_max = min(w - 1, x + int(width_px))
            y_max = min(h - 1, y + int(height_px))

            valid_width = x_max - x_min
            valid_height = y_max - y_min

            if valid_width <= 0 or valid_height <= 0:
                logging.warning(f"跳过无效边界框 {bbox_cv} (太小或超出图像范围)。")
                continue

            bbox_cv_valid = (x_min, y_min, valid_width, valid_height)

            tracker = self.tracker_creator()
            try:
                success = tracker.init(frame_rgb, bbox_cv_valid)
                if success:
                    self.trackers.append((tracker, ann.class_id, ann.track_id, ann.track_name))
                else:
                    logging.error(f"无法初始化跟踪器，边界框: {bbox_cv_valid}")
            except Exception as e:
                logging.error(f"初始化跟踪器时发生异常，边界框: {bbox_cv_valid}, 错误: {e}")

    def update_trackers(self, next_frame_rgb: np.array) -> list:
        predicted_annotations = []
        if not self.trackers:
            return predicted_annotations

        h, w, _ = next_frame_rgb.shape

        active_trackers = []
        for i, (tracker, class_id, track_id, track_name) in enumerate(self.trackers):
            if tracker is not None:
                try:
                    success, bbox_cv = tracker.update(next_frame_rgb)
                    if success:
                        x_center_px = bbox_cv[0] + bbox_cv[2] / 2
                        y_center_px = bbox_cv[1] + bbox_cv[3] / 2

                        normalized_x_center = np.clip(x_center_px / w, 0.0, 1.0)
                        normalized_y_center = np.clip(y_center_px / h, 0.0, 1.0)
                        normalized_width = np.clip(bbox_cv[2] / w, 0.0, 1.0)
                        normalized_height = np.clip(bbox_cv[3] / h, 0.0, 1.0)

                        min_dim_ratio = 5.0 / max(w, h)
                        normalized_width = max(normalized_width, min_dim_ratio)
                        normalized_height = max(normalized_height, min_dim_ratio)

                        if normalized_x_center - normalized_width / 2 < 0:
                            normalized_x_center = normalized_width / 2
                        elif normalized_x_center + normalized_width / 2 > 1:
                            normalized_x_center = 1 - normalized_width / 2

                        if normalized_y_center - normalized_height / 2 < 0:
                            normalized_y_center = normalized_height / 2
                        elif normalized_y_center + normalized_height / 2 > 1:
                            normalized_y_center = 1 - normalized_height / 2

                        new_bbox = BBox(normalized_x_center, normalized_y_center, normalized_width, normalized_height)

                        predicted_annotations.append(
                            Annotation(class_id, new_bbox, track_id=track_id, track_name=track_name))
                        active_trackers.append((tracker, class_id, track_id, track_name))
                    else:
                        logging.warning(f"跟踪器 {i} 在下一帧中丢失目标。")
                except Exception as e:
                    logging.error(f"更新跟踪器 {i} 时发生异常: {e}")

        self.trackers = active_trackers

        return predicted_annotations

    def reset(self):
        self.trackers = []