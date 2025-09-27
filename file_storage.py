import os
import shutil
import zipfile
import yaml
import random
import cv2
import numpy as np

import config
from bbox_writer import convert_to_yolo_format, convert_text_to_rects_and_labels


def init_storage():
    os.makedirs(config.STORAGE_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.STORAGE_DIR, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(config.STORAGE_DIR, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(config.STORAGE_DIR, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(config.STORAGE_DIR, 'models'), exist_ok=True)


def get_video_path(video_uuid):
    return os.path.join(config.STORAGE_DIR, 'videos', f"{video_uuid}.mp4")


def save_uploaded_video(file_storage_obj, video_uuid):
    video_path = get_video_path(video_uuid)
    file_storage_obj.save(video_path)
    return video_path


def delete_video_file(video_uuid):
    video_path = get_video_path(video_uuid)
    if os.path.exists(video_path):
        os.remove(video_path)


def get_frame_dir(video_uuid):
    return os.path.join(config.STORAGE_DIR, 'frames', video_uuid)


def get_frame_path(video_uuid, frame_number):
    frame_dir = get_frame_dir(video_uuid)
    return os.path.join(frame_dir, f"frame_{frame_number:05d}.jpg")


def save_frame_image(video_uuid, frame_number, image_data_bytes):
    frame_dir = get_frame_dir(video_uuid)
    os.makedirs(frame_dir, exist_ok=True)
    frame_path = get_frame_path(video_uuid, frame_number)
    with open(frame_path, 'wb') as f:
        f.write(image_data_bytes)


def delete_frames_for_video(video_uuid):
    frame_dir = get_frame_dir(video_uuid)
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)


def get_dataset_dir(dataset_uuid):
    return os.path.join(config.STORAGE_DIR, 'datasets', dataset_uuid)


def get_dataset_zip_path(dataset_uuid):
    return os.path.join(config.STORAGE_DIR, 'datasets', f"{dataset_uuid}.zip")


def get_yolo_bboxes(bboxes_text, width, height, class_map):
    rects, labels, _ = convert_text_to_rects_and_labels(bboxes_text)
    if not rects: return [], []

    yolo_bboxes = []
    class_indices = []

    for i, r in enumerate(rects):
        x1, y1, x2, y2 = r

        norm_x1 = x1 / width
        norm_y1 = y1 / height
        norm_x2 = x2 / width
        norm_y2 = y2 / height
        norm_x1 = max(0.0, min(1.0, norm_x1))
        norm_y1 = max(0.0, min(1.0, norm_y1))
        norm_x2 = max(0.0, min(1.0, norm_x2))
        norm_y2 = max(0.0, min(1.0, norm_y2))

        box_w = norm_x2 - norm_x1
        box_h = norm_y2 - norm_y1
        x_center = norm_x1 + box_w / 2
        y_center = norm_y1 + box_h / 2

        if box_w > 0 and box_h > 0:
            yolo_bboxes.append([x_center, y_center, box_w, box_h])
            class_indices.append(class_map[labels[i]])

    return yolo_bboxes, class_indices


def create_mosaic_image(image_infos, class_map):
    output_dim = max(info['width'] for info in image_infos)
    s = output_dim
    yc, xc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]

    mosaic_border = [-s // 2, -s // 2]
    mosaic_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
    final_bboxes = []

    for i, info in enumerate(image_infos):
        img = cv2.imread(get_frame_path(info['video_uuid'], info['frame_number']))
        h, w, _ = img.shape

        if i == 0:
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        yolo_boxes, class_indices = get_yolo_bboxes(info['bboxes_text'], w, h, class_map)

        for j, box in enumerate(yolo_boxes):
            x_center, y_center, box_w, box_h = box

            x_center_abs, box_w_abs = x_center * w, box_w * w
            y_center_abs, box_h_abs = y_center * h, box_h * h

            new_x_center = (x_center_abs + padw) / (s * 2)
            new_y_center = (y_center_abs + padh) / (s * 2)
            new_w = box_w_abs / (s * 2)
            new_h = box_h_abs / (s * 2)

            final_bboxes.append((class_indices[j], new_x_center, new_y_center, new_w, new_h))

    return mosaic_img, final_bboxes


def create_yolo_dataset_zip(dataset_uuid, frames_data, all_labels, eval_percent, test_percent, augment_pipeline=None,
                            mosaic_options=None):
    dataset_dir = get_dataset_dir(dataset_uuid)
    if os.path.exists(dataset_dir): shutil.rmtree(dataset_dir)
    img_train_dir = os.path.join(dataset_dir, 'images', 'train')
    lbl_train_dir = os.path.join(dataset_dir, 'labels', 'train')
    img_val_dir = os.path.join(dataset_dir, 'images', 'val')
    lbl_val_dir = os.path.join(dataset_dir, 'labels', 'val')
    img_test_dir = os.path.join(dataset_dir, 'images', 'test')
    lbl_test_dir = os.path.join(dataset_dir, 'labels', 'test')
    os.makedirs(img_train_dir);
    os.makedirs(lbl_train_dir)
    os.makedirs(img_val_dir);
    os.makedirs(lbl_val_dir)
    os.makedirs(img_test_dir);
    os.makedirs(lbl_test_dir)

    class_map = {name: i for i, name in enumerate(all_labels)}
    random.shuffle(frames_data)
    total_count = len(frames_data)
    val_count = int(total_count * eval_percent / 100.0)
    test_count = int(total_count * test_percent / 100.0)
    val_data = frames_data[:val_count]
    test_data = frames_data[val_count:val_count + test_count]
    train_data = frames_data[val_count + test_count:]

    dataset_parts = [(train_data, img_train_dir, lbl_train_dir), (val_data, img_val_dir, lbl_val_dir),
                     (test_data, img_test_dir, lbl_test_dir)]

    for part_data, target_img_dir, target_lbl_dir in dataset_parts:
        is_training_set = (target_img_dir == img_train_dir)

        while part_data:
            use_mosaic = is_training_set and mosaic_options.get('enabled') and random.random() < mosaic_options.get('p',
                                                                                                                    0) and len(
                part_data) >= 4

            if use_mosaic:
                mosaic_infos = [part_data.pop(random.randrange(len(part_data))) for _ in range(4)]
                base_filename = f"mosaic_{mosaic_infos[0]['video_uuid']}_{mosaic_infos[0]['frame_number']}"
                image_aug, bboxes_aug_yolo = create_mosaic_image(mosaic_infos, class_map)
                final_image_bgr = image_aug
            else:
                frame_info = part_data.pop(0)
                is_augmented = frame_info.get("type") == "augmented"
                if is_augmented:
                    base_filename = frame_info["augmented_id"]
                else:
                    base_filename = f"{frame_info['video_uuid']}_{frame_info['frame_number']:05d}"

                src_img_path = get_frame_path(frame_info['video_uuid'], frame_info['frame_number'])
                if not os.path.exists(src_img_path): continue

                image = cv2.imread(src_img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                yolo_bboxes, class_indices = get_yolo_bboxes(frame_info['bboxes_text'], frame_info['width'],
                                                             frame_info['height'], class_map)

                if not yolo_bboxes: continue

                if is_augmented and augment_pipeline:
                    transformed = augment_pipeline(image=image, bboxes=yolo_bboxes, class_labels=class_indices)
                    image_aug_rgb = transformed['image']
                    bboxes_aug_yolo_tuples = transformed['bboxes']
                    labels_aug_indices = transformed['class_labels']
                    bboxes_aug_yolo = [(labels_aug_indices[i], *box) for i, box in enumerate(bboxes_aug_yolo_tuples)]
                else:
                    image_aug_rgb = image
                    bboxes_aug_yolo = [(class_indices[i], *box) for i, box in enumerate(yolo_bboxes)]

                final_image_bgr = cv2.cvtColor(image_aug_rgb, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(target_img_dir, base_filename + '.jpg'), final_image_bgr)
            yolo_content_lines = [f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for class_id, x, y, w, h in
                                  bboxes_aug_yolo]
            with open(os.path.join(target_lbl_dir, base_filename + '.txt'), 'w') as f:
                f.write("\n".join(yolo_content_lines))

    yaml_content = {'path': f"../datasets/{dataset_uuid}", 'train': 'images/train', 'val': 'images/val',
                    'test': 'images/test', 'nc': len(all_labels), 'names': all_labels}
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    zip_path = get_dataset_zip_path(dataset_uuid)
    shutil.make_archive(os.path.join(config.STORAGE_DIR, 'datasets', dataset_uuid), 'zip', dataset_dir)
    shutil.rmtree(dataset_dir)
    return zip_path


def delete_dataset_files(dataset_uuid):
    dataset_dir = get_dataset_dir(dataset_uuid)
    zip_path = get_dataset_zip_path(dataset_uuid)
    if os.path.isdir(dataset_dir): shutil.rmtree(dataset_dir)
    if os.path.exists(zip_path): os.remove(zip_path)


def get_model_path(model_uuid):
    return os.path.join(config.STORAGE_DIR, 'models', f"{model_uuid}.tflite")


def get_label_file_path(model_uuid):
    return os.path.join(config.STORAGE_DIR, 'models', f"{model_uuid}.txt")


def save_imported_model(file_storage_obj, model_uuid):
    model_path = get_model_path(model_uuid)
    file_storage_obj.save(model_path)
    return model_path


def save_imported_label_file(file_storage_obj, model_uuid):
    label_path = get_label_file_path(model_uuid)
    file_storage_obj.save(label_path)
    return label_path


def delete_model_file(model_uuid):
    model_path = get_model_path(model_uuid)
    if os.path.exists(model_path): os.remove(model_path)


def delete_label_file(model_uuid):
    label_path = get_label_file_path(model_uuid)
    if os.path.exists(label_path): os.remove(label_path)