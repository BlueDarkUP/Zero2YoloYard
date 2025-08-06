# sam_worker.py

from PyQt5.QtCore import QObject, pyqtSignal as Signal
import numpy as np

class SamWorker(QObject):
    finished = Signal(np.ndarray)
    error = Signal(str)

    def __init__(self, model_manager, image_np, point_coords, point_labels):
        super(SamWorker, self).__init__()
        self.model_manager = model_manager
        self.image_np = image_np
        self.point_coords = point_coords
        self.point_labels = point_labels
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        if not self.is_running:
            return
        try:
            mask = self.model_manager.run_sam_inference(
                self.image_np, self.point_coords, self.point_labels
            )
            if mask is not None and self.is_running:
                self.finished.emit(mask)
            elif self.is_running:
                self.error.emit("SAM未能生成有效的掩码。")
        except Exception as e:
            import traceback
            self.error.emit(f"SAM推理失败: {e}\n{traceback.format_exc()}")