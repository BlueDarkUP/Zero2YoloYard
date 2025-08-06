# model_diagnosis_worker.py
from PySide6.QtCore import QObject, Signal
from ai_models import AI_AVAILABLE, ModelManager, run_yolo_diagnosis


def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


class DiagnosisWorker(QObject):
    finished = Signal(dict)
    log = Signal(str)

    def __init__(self, all_image_annotations, model_name='yolov8n'):
        super().__init__()
        if not AI_AVAILABLE: raise ImportError("需要AI库才能进行模型诊断。")
        self.all_image_annotations = all_image_annotations
        self.model_name = model_name

    def run(self):
        try:
            self.log.emit("模型诊断开始...")
            model = ModelManager(self.model_name)
            if model is None:
                self.finished.emit({"error": "模型加载失败。"})
                return

            potential_misses = {}
            potential_fps = {}
            class_mismatches = {}

            for i, image_ann in enumerate(self.all_image_annotations):
                self.log.emit(f"正在诊断图像 {i + 1}/{len(self.all_image_annotations)}...")

                gt_boxes = []
                for ann in image_ann.annotations:
                    w_px, h_px = ann.bbox.width * image_ann.width, ann.bbox.height * image_ann.height
                    x1 = (ann.bbox.x_center * image_ann.width) - w_px / 2
                    y1 = (ann.bbox.y_center * image_ann.height) - h_px / 2
                    gt_boxes.append([x1, y1, x1 + w_px, y1 + h_px, ann.class_id])

                preds = run_yolo_diagnosis(model, image_ann.image_path)

                gt_matched = [False] * len(gt_boxes)
                pred_matched = [False] * len(preds)

                for p_idx, pred in enumerate(preds):
                    for gt_idx, gt in enumerate(gt_boxes):
                        if box_iou(pred['box'], gt) > 0.5:
                            gt_matched[gt_idx] = True
                            pred_matched[p_idx] = True
                            if pred['class_id'] != gt[4]:
                                if i not in class_mismatches: class_mismatches[i] = []
                                class_mismatches[i].append({'gt': gt, 'pred': pred})
                            break

                for p_idx, pred in enumerate(preds):
                    if not pred_matched[p_idx]:
                        if i not in potential_misses: potential_misses[i] = []
                        potential_misses[i].append(pred)

                for gt_idx, gt in enumerate(gt_boxes):
                    if not gt_matched[gt_idx]:
                        if i not in potential_fps: potential_fps[i] = []
                        potential_fps[i].append(gt)

            results = {
                "potential_misses": potential_misses,
                "potential_fps": potential_fps,
                "class_mismatches": class_mismatches
            }
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.finished.emit({"error": f"{e}\n{traceback.format_exc()}"})