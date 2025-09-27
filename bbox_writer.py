# bbox_writer.py

import logging
import numpy as np
import config


def __convert_bbox_to_text(bbox, scale, x_max, y_max):
    p0 = bbox[:2].astype(float)
    p1 = p0 + bbox[2:].astype(float)
    size = p1 - p0
    center = p0 + (size / 2)
    new_size = scale * size
    p0 = center - new_size / 2
    p1 = center + new_size / 2
    scaled_bbox = np.array([p0, p1 - p0]).reshape(-1)
    p0 = scaled_bbox[:2]
    size = scaled_bbox[2:]
    p1 = p0 + size
    return "%d,%d,%d,%d" % (
        int(max(p0[0], 0)),
        int(max(p0[1], 0)),
        int(min(p1[0], x_max)),
        int(min(p1[1], y_max)))


def __convert_bboxes_and_labels_to_text(bboxes, scale, max_x, max_y, labels):
    assert (len(bboxes) == len(labels))
    bboxes_text = ""
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        label = labels[i]
        if bbox is None or label is None:
            continue
        bboxes_text += "%s,%s\n" % (__convert_bbox_to_text(bbox, scale, max_x, max_y), label)
    return bboxes_text


def __convert_rects_to_bboxes(rects):
    bboxes = []
    for rect in rects:
        p0 = rect[:2]
        p1 = rect[2:]
        size = p1 - p0
        bbox = np.array([p0, size]).reshape(-1)
        bboxes.append(bbox)
    return bboxes


def validate_bboxes_text(s):
    if s is None:
        return ""
    lines = s.split("\n")
    for line in lines:
        if len(line.strip()) > 0:
            try:
                parts = line.strip().split(',', 4)
                if len(parts) != 5:
                    raise ValueError("Line does not have enough parts for rect and label.")
                rect_str = parts[:4]
                np.array(rect_str, dtype=float).astype(int)
            except Exception as e:
                message = f"Error: Line '{line}' is not a valid bbox format. Details: {e}"
                logging.critical(message)
                raise ValueError(message)
    return s


def convert_text_to_rects_and_labels(bboxes_text):
    """
    稳健地将多行边界框文本解析为矩形和标签列表。
    使用 split(',', 4) 来确保标签内的逗号不会导致解析错误。
    新增：自动纠正反向绘制的框 (x1 > x2 或 y1 > y2)。
    """
    rects = []
    labels = []
    if not bboxes_text:
        return rects, labels

    lines = [line for line in bboxes_text.split('\n') if line.strip()]
    for line in lines:
        try:
            parts = line.strip().split(',', 4)
            if len(parts) != 5:
                logging.warning(f"Skipping malformed bbox line (not enough parts): '{line}'")
                continue

            rect_str = parts[:4]
            label = parts[4]

            coords = np.array(rect_str, dtype=float).astype(int)

            # --- MODIFICATION START ---
            # 确保 x1 < x2 and y1 < y2，纠正反向绘制的框
            x1, y1, x2, y2 = coords
            rect = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
            # --- MODIFICATION END ---

            rects.append(rect)
            labels.append(label)
        except (ValueError, IndexError) as e:
            logging.warning(f"Skipping malformed bbox line (parsing error): '{line}'. Error: {e}")
            continue
    return rects, labels


def count_boxes(bboxes_text):
    if not bboxes_text:
        return 0
    return len([line for line in bboxes_text.split('\n') if line.strip()])


def __convert_text_to_bboxes_and_labels(bboxes_text):
    rects, labels = convert_text_to_rects_and_labels(bboxes_text)
    bboxes = __convert_rects_to_bboxes(rects)
    return bboxes, labels


def __scale_bboxes(bboxes, scale):
    scaled_bboxes = []
    for bbox in bboxes:
        if bbox is None:
            scaled_bboxes.append(None)
        else:
            p0 = bbox[:2].astype(float)
            p1 = p0 + bbox[2:].astype(float)
            size = p1 - p0
            center = p0 + (size / 2)
            new_size = scale * size
            p0 = center - new_size / 2
            p1 = center + new_size / 2
            scaled_bboxes.append(np.array([p0, p1 - p0]).reshape(-1))
    return scaled_bboxes


def parse_bboxes_text(bboxes_text, scale=1):
    bboxes_, labels = __convert_text_to_bboxes_and_labels(bboxes_text)
    bboxes = __scale_bboxes(bboxes_, scale)
    return bboxes, labels


def extract_labels(bboxes_text):
    _, labels = convert_text_to_rects_and_labels(bboxes_text)
    return labels


def format_bboxes_text(bboxes, labels, scale, max_x, max_y):
    return __convert_bboxes_and_labels_to_text(bboxes, 1 / scale, max_x, max_y, labels)


def convert_to_yolo_format(bboxes_text, class_map, image_width, image_height):
    rects, labels = convert_text_to_rects_and_labels(bboxes_text)
    yolo_lines = []
    for i, rect in enumerate(rects):
        label = labels[i]
        if label not in class_map: continue
        class_id = class_map[label]
        x1, y1, x2, y2 = rect
        box_width = float(x2 - x1)
        box_height = float(y2 - y1)
        x_center = float(x1) + (box_width / 2)
        y_center = float(y1) + (box_height / 2)
        x_center_norm = x_center / image_width
        y_center_norm = y_center / image_height
        width_norm = box_width / image_width
        height_norm = box_height / image_height
        yolo_lines.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
    return "\n".join(yolo_lines)