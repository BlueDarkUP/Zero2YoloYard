import os
import shutil
import zipfile
import yaml
import random

import config
from bbox_writer import convert_to_yolo_format


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


def create_yolo_dataset_zip(dataset_uuid, frames_data, all_labels, eval_percent, test_percent):
    dataset_dir = get_dataset_dir(dataset_uuid)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    img_train_dir = os.path.join(dataset_dir, 'images', 'train')
    lbl_train_dir = os.path.join(dataset_dir, 'labels', 'train')
    img_val_dir = os.path.join(dataset_dir, 'images', 'val')
    lbl_val_dir = os.path.join(dataset_dir, 'labels', 'val')
    img_test_dir = os.path.join(dataset_dir, 'images', 'test')
    lbl_test_dir = os.path.join(dataset_dir, 'labels', 'test')

    os.makedirs(img_train_dir)
    os.makedirs(lbl_train_dir)
    os.makedirs(img_val_dir)
    os.makedirs(lbl_val_dir)
    os.makedirs(img_test_dir)
    os.makedirs(lbl_test_dir)

    class_map = {name: i for i, name in enumerate(all_labels)}
    random.shuffle(frames_data)

    total_count = len(frames_data)
    val_count = int(total_count * eval_percent / 100.0)
    test_count = int(total_count * test_percent / 100.0)

    # 从列表开头取 val 和 test 数据，剩下的是 train 数据
    val_data = frames_data[:val_count]
    test_data = frames_data[val_count:val_count + test_count]
    train_data = frames_data[val_count + test_count:]

    dataset_parts = [
        (train_data, img_train_dir, lbl_train_dir),
        (val_data, img_val_dir, lbl_val_dir),
        (test_data, img_test_dir, lbl_test_dir)
    ]
    for dataset_part, target_img_dir, target_lbl_dir in dataset_parts:
        for video_uuid, frame_num, bboxes, width, height in dataset_part:
            base_filename = f"{video_uuid}_{frame_num:05d}"
            src_img_path = get_frame_path(video_uuid, frame_num)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, os.path.join(target_img_dir, base_filename + '.jpg'))
                yolo_content = convert_to_yolo_format(bboxes, class_map, width, height)
                with open(os.path.join(target_lbl_dir, base_filename + '.txt'), 'w') as f:
                    f.write(yolo_content)

    yaml_content = {
        'path': f"../datasets/{dataset_uuid}",
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(all_labels),
        'names': all_labels
    }
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    zip_path = get_dataset_zip_path(dataset_uuid)
    shutil.make_archive(os.path.join(config.STORAGE_DIR, 'datasets', dataset_uuid), 'zip', dataset_dir)
    shutil.rmtree(dataset_dir)

    return zip_path


def delete_dataset_files(dataset_uuid):
    dataset_dir = get_dataset_dir(dataset_uuid)
    zip_path = get_dataset_zip_path(dataset_uuid)
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)


def get_model_path(model_uuid):
    return os.path.join(config.STORAGE_DIR, 'models', f"{model_uuid}.tflite")


def save_imported_model(file_storage_obj, model_uuid):
    model_path = get_model_path(model_uuid)
    file_storage_obj.save(model_path)
    return model_path


def delete_model_file(model_uuid):
    model_path = get_model_path(model_uuid)
    if os.path.exists(model_path):
        os.remove(model_path)