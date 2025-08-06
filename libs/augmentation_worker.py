import os
import cv2
import numpy as np
import random
import time
import traceback
from PyQt5.QtCore import QObject, pyqtSignal as Signal
import albumentations as A
from formats.internal_data import ImageAnnotation, Annotation, BBox
import concurrent.futures


def process_single_item(args):
    image_ann, config = args
    base_name = os.path.splitext(os.path.basename(image_ann.image_path))[0]

    results_for_item = []

    if config['copy_originals']:
        results_for_item.append(('original', base_name, image_ann.image_path, image_ann))

    if config['num_augs'] > 0 and config['has_standard_augs']:
        if hasattr(image_ann, 'image_data') and image_ann.image_data is not None:
            image_rgb = image_ann.image_data
        else:
            image_rgb = cv2.cvtColor(cv2.imread(image_ann.image_path), cv2.COLOR_BGR2RGB)

        if image_rgb is None:
            return []

        bboxes = [[ann.bbox.x_center, ann.bbox.y_center, ann.bbox.width, ann.bbox.height] for ann in
                  image_ann.annotations]
        class_ids = [ann.class_id for ann in image_ann.annotations]

        for j in range(config['num_augs']):
            geo_augmented = config['geo_transform'](image=image_rgb, bboxes=bboxes, class_labels=class_ids)
            if not geo_augmented['bboxes']: continue
            final_image = config['pixel_transform'](image=geo_augmented['image'])['image']

            new_annotations = [Annotation(cid, BBox(*b_vals)) for b_vals, cid in
                               zip(geo_augmented['bboxes'], geo_augmented['class_labels'])]
            aug_ann = ImageAnnotation(image_ann.image_path, final_image.shape[1], final_image.shape[0], new_annotations)
            aug_ann.image_data = final_image  # Attach augmented image data for writer

            results_for_item.append(('augmented', f"{base_name}_aug_{j}", final_image, aug_ann))

    return results_for_item


class AugmentationWorker(QObject):
    progress = Signal(int)
    finished = Signal(str)
    log = Signal(str)

    def __init__(self, config, readers_map, writers_map):
        super().__init__()
        self.config = config;
        self.readers_map = readers_map;
        self.writers_map = writers_map
        self._is_running = True

    def stop(self):
        self._is_running = False;
        self.log.emit("正在请求停止任务...")
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False, cancel_futures=True)

    def run(self):
        writer = None
        try:
            self.log.emit("任务开始...")

            ReaderClass = self.readers_map[self.config['input_format']]
            reader = ReaderClass(self.config['dataset_path'])
            all_image_annotations, class_names = reader.read()
            if not all_image_annotations:
                self.finished.emit("任务中止: 未能加载任何标注信息。");
                return
            self.log.emit(f"加载完成！共 {len(all_image_annotations)} 张图像。")

            if self.config['preload_images']:
                self.log.emit("正在预加载所有图像到内存中...")
                for i, ann_obj in enumerate(all_image_annotations):
                    ann_obj.image_data = cv2.cvtColor(cv2.imread(ann_obj.image_path), cv2.COLOR_BGR2RGB)
                    if i % 100 == 0: self.log.emit(f"已加载 {i + 1}/{len(all_image_annotations)}...")
                self.log.emit("图像预加载完成！")
            else:
                for ann_obj in all_image_annotations:
                    if hasattr(ann_obj, 'image_data'):
                        delattr(ann_obj, 'image_data')

            WriterClass = self.writers_map[self.config['output_format']]
            output_path = os.path.join(self.config['dataset_path'],
                                       f"processed_{self.config['output_format'].replace(' ', '_')}")
            writer = WriterClass(output_path, class_names)

            geometric_augs, pixel_augs = [], []
            for aug_config in self.config['augmentations'].values():
                if aug_config['enabled']:
                    aug_type = aug_config['config'].get('type', 'pixel')
                    instance = aug_config['class'](**aug_config['values'])
                    if aug_type == 'geometric':
                        geometric_augs.append(instance)
                    else:
                        pixel_augs.append(instance)

            self.config['geo_transform'] = A.Compose(geometric_augs, bbox_params=A.BboxParams(format='yolo',
                                                                                              label_fields=[
                                                                                                  'class_labels'],
                                                                                              min_visibility=0.1))
            self.config['pixel_transform'] = A.Compose(pixel_augs)
            self.config['has_standard_augs'] = bool(geometric_augs or pixel_augs)

            self.log.emit(f"阶段 1/2: 使用 {self.config['num_workers']} 个核心进行标准处理...")
            processed_count = 0;
            total_to_process = len(all_image_annotations)
            tasks = [(ann, self.config) for ann in all_image_annotations]

            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.config['num_workers'])
            futures = {self.executor.submit(process_single_item, task) for task in tasks}

            for future in concurrent.futures.as_completed(futures):
                if not self._is_running: break
                item_results = future.result()
                for res_type, basename, data, ann in item_results:
                    writer.write(data, ann, basename)
                processed_count += 1
                self.progress.emit(int((processed_count / total_to_process) * 50))

            if not self._is_running: raise InterruptedError("任务被用户中止。")

            if self.config['mosaic_enabled']:
                self.log.emit("阶段 2/2: 正在生成 Mosaic 图像...")
                num_mosaic = self.config['num_mosaic']
                for i in range(num_mosaic):
                    if not self._is_running: break
                    self.progress.emit(50 + int(((i + 1) / num_mosaic) * 50))
                    self._generate_one_mosaic(all_image_annotations, writer, f"mosaic_{i}")
            else:
                self.progress.emit(100)

            if not self._is_running: raise InterruptedError("任务被用户中止。")

        except (InterruptedError, concurrent.futures.CancelledError) as e:
            self.log.emit("任务已被用户中止。")
        except Exception as e:
            self.log.emit(f"发生严重错误: {e}\n{traceback.format_exc()}");
            self.finished.emit(f"发生严重错误: {e}")
        finally:
            if self._is_running:
                if writer and hasattr(writer, 'finalize') and callable(getattr(writer, 'finalize')):
                    try:
                        writer.finalize();
                        self.log.emit("最终文件写入完成。")
                    except Exception as e:
                        self.log.emit(f"终结步骤发生错误: {e}")
            else:
                self.log.emit("任务被取消，跳过最终文件写入。")

            self.finished.emit("所有任务已完成！" if self._is_running else "任务已被用户中止。")

    def _generate_one_mosaic(self, dataset, writer, new_basename):
        samples = random.sample(dataset, 4);
        output_dim = 1280
        canvas = np.full((output_dim, output_dim, 3), 114, dtype=np.uint8)
        center_x, center_y = output_dim // 2, output_dim // 2
        mosaic_annotations = []
        for i, sample in enumerate(samples):
            if hasattr(sample, 'image_data') and sample.image_data is not None:
                img = sample.image_data
            else:
                img = cv2.cvtColor(cv2.imread(sample.image_path), cv2.COLOR_BGR2RGB)

            if img is None: continue

            h, w = img.shape[:2];
            scale = random.uniform(0.5, 1.5)
            img = cv2.resize(img, (int(w * scale), int(h * scale)));
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(center_x - w, 0), max(center_y - h, 0), center_x, center_y
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = center_x, max(center_y - h, 0), min(center_x + w, output_dim), center_y
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(center_x - w, 0), center_y, center_x, min(center_y + h, output_dim)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            else:
                x1a, y1a, x2a, y2a = center_x, center_y, min(center_x + w, output_dim), min(center_y + h, output_dim)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_x = x1a - x1b;
            pad_y = y1a - y1b
            for ann in sample.annotations:
                x_center_new = (ann.bbox.x_center * sample.width) * scale + pad_x
                y_center_new = (ann.bbox.y_center * sample.height) * scale + pad_y
                bbox_w_scaled = ann.bbox.width * sample.width * scale
                bbox_h_scaled = ann.bbox.height * sample.height * scale
                x_min_new = np.clip(x_center_new - bbox_w_scaled / 2, 0, output_dim)
                y_min_new = np.clip(y_center_new - bbox_h_scaled / 2, 0, output_dim)
                x_max_new = np.clip(x_center_new + bbox_w_scaled / 2, 0, output_dim)
                y_max_new = np.clip(y_center_new + bbox_h_scaled / 2, 0, output_dim)
                if x_max_new - x_min_new > 1 and y_max_new - y_min_new > 1:
                    new_w = (x_max_new - x_min_new) / output_dim;
                    new_h = (y_max_new - y_min_new) / output_dim
                    new_xc = (x_max_new + x_min_new) / 2 / output_dim;
                    new_yc = (y_max_new + y_min_new) / 2 / output_dim
                    mosaic_annotations.append(Annotation(ann.class_id, BBox(new_xc, new_yc, new_w, new_h)))
        if mosaic_annotations:
            mosaic_image_ann = ImageAnnotation(f"{new_basename}.jpg", output_dim, output_dim, mosaic_annotations)
            writer.write(canvas, mosaic_image_ann, new_basename)