# local_tracker_manager.py
import cv2
import numpy as np
import logging


# --- 新增的辅助函数，用于兼容不同OpenCV版本 ---
def create_tracker_by_name(tracker_type: str):
    """
    动态地创建OpenCV跟踪器实例，处理版本兼容性问题。
    它会先尝试从 cv2.legacy 模块创建，如果失败，再尝试从主 cv2 模块创建。
    """
    tracker_creators = {
        'CSRT': ['TrackerCSRT_create', 'TrackerCSRT_create'],
        'KCF': ['TrackerKCF_create', 'TrackerKCF_create'],
        'MIL': ['TrackerMIL_create', 'TrackerMIL_create'],
        'Boosting': ['TrackerBoosting_create', 'TrackerBoosting_create'],
        'MedianFlow': ['TrackerMedianFlow_create', 'TrackerMedianFlow_create'],
        'MOSSE': ['TrackerMOSSE_create', 'TrackerMOSSE_create'],
        'TLD': ['TrackerTLD_create', 'TrackerTLD_create'],
    }

    if tracker_type not in tracker_creators:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}")

    creator_func_name = tracker_creators[tracker_type][0]

    # 优先尝试 cv2.legacy (适用于OpenCV 4.5.1+ 的一些版本)
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, creator_func_name):
            return getattr(cv2.legacy, creator_func_name)()
    except Exception:
        pass  # 如果失败，则尝试下一个

    # 备选方案，尝试主 cv2 模块
    try:
        if hasattr(cv2, creator_func_name):
            return getattr(cv2, creator_func_name)()
    except Exception:
        pass  # 如果失败，则抛出最终错误

    # 如果两个位置都找不到
    raise ImportError(
        f"无法创建跟踪器 '{tracker_type}'。\n"
        f"请确保您已正确安装 'opencv-contrib-python' 并且版本兼容。\n"
        f"请运行: 'pip install --upgrade opencv-contrib-python'"
    )


class LocalTrackerManager:
    def __init__(self, tracker_type: str = "CSRT"):
        self.tracker_type = tracker_type
        self.trackers = []
        # 创建函数的调用被移到了 init_trackers 中

    def _shape_to_cv_bbox(self, shape, img_width, img_height):
        points = [(p.x(), p.y()) for p in shape.points]
        x_min = min(p[0] for p in points)
        y_min = min(p[1] for p in points)
        x_max = max(p[0] for p in points)
        y_max = max(p[1] for p in points)

        x = int(x_min)
        y = int(y_min)
        w = int(x_max - x_min)
        h = int(y_max - y_min)

        x = max(0, x)
        y = max(0, y)
        w = min(w, img_width - x)
        h = min(h, img_height - y)

        return x, y, w, h

    def init_trackers(self, frame_bgr, shapes):
        self.trackers = []
        if not shapes:
            logging.warning("没有提供标注用于初始化跟踪器。")
            return 0

        if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
            logging.error("跟踪器需要一个3通道的彩色图像。")
            return 0
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        h, w, _ = frame_bgr.shape
        successful_inits = 0

        for shape in shapes:
            bbox_cv = self._shape_to_cv_bbox(shape, w, h)
            if bbox_cv[2] <= 0 or bbox_cv[3] <= 0:
                logging.warning(f"跳过无效边界框 {bbox_cv} (尺寸为0或负)。")
                continue

            # 使用新的辅助函数来创建跟踪器
            tracker = create_tracker_by_name(self.tracker_type)

            try:
                success = tracker.init(frame_bgr, bbox_cv)
                if success:
                    self.trackers.append({'tracker': tracker, 'label': shape.label})
                    successful_inits += 1
                else:
                    logging.error(f"无法初始化跟踪器 (标签: {shape.label})，边界框: {bbox_cv}。")
            except Exception as e:
                logging.error(f"初始化跟踪器时发生异常 (标签: {shape.label})，边界框: {bbox_cv}, 错误: {e}")

        return successful_inits

    def update_trackers(self, next_frame_bgr):
        predicted_bboxes = []
        if not self.trackers:
            return predicted_bboxes

        if len(next_frame_bgr.shape) != 3 or next_frame_bgr.shape[2] != 3:
            logging.error("跟踪器需要一个3通道的彩色图像。")
            return []
        if next_frame_bgr.dtype != np.uint8:
            next_frame_bgr = next_frame_bgr.astype(np.uint8)

        active_trackers = []
        for tracker_info in self.trackers:
            tracker = tracker_info['tracker']
            try:
                success, bbox_cv = tracker.update(next_frame_bgr)
                if success:
                    predicted_bboxes.append({'bbox': bbox_cv, 'label': tracker_info['label']})
                    active_trackers.append(tracker_info)
                else:
                    logging.warning("跟踪器丢失目标。")
            except Exception as e:
                logging.error(f"更新跟踪器时发生异常: {e}")

        self.trackers = active_trackers
        return predicted_bboxes

    def reset(self):
        self.trackers = []