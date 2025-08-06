# split_worker.py
import os
import random
import shutil
from collections import defaultdict
from PySide6.QtCore import QObject, Signal


class SplitWorker(QObject):
    finished = Signal(str)
    log = Signal(str)
    progress = Signal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.log.emit("数据集分割任务开始...")
            self.progress.emit(0)
            source_dir = self.config['source_dir']

            # --- 核心修改部分开始 ---
            image_dir = os.path.join(source_dir, 'images')
            label_dir = os.path.join(source_dir, 'labels')

            # 检查标准 'images'/'labels' 子目录结构
            if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
                self.log.emit(f"未找到 'images'/'labels' 子目录，将尝试直接从源目录 '{source_dir}' 读取文件。")
                # 如果标准结构不存在，则假定源目录本身就是图片和标签的存放地
                image_dir = source_dir
                label_dir = source_dir
            # --- 核心修改部分结束 ---

            # 现在，我们扫描 image_dir 来获取所有可能的图片文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            try:
                all_files = os.listdir(image_dir)
                basenames = [os.path.splitext(f)[0] for f in all_files if
                             os.path.splitext(f)[1].lower() in image_extensions]
            except FileNotFoundError:
                self.finished.emit(f"错误: 在目录 '{image_dir}' 中找不到任何文件。")
                return

            if not basenames:
                self.finished.emit(f"错误: 在目录 '{image_dir}' 中没有找到任何支持的图片文件。")
                return

            random.shuffle(basenames)
            self.progress.emit(5)

            if self.config['stratified']:
                self.log.emit("使用分层采样进行分割...")
                train_files, valid_files, test_files = self._stratified_split(label_dir, basenames)
            else:
                self.log.emit("使用随机采样进行分割...")
                train_files, valid_files, test_files = self._random_split(basenames)

            self.progress.emit(10)
            self.log.emit(f"分割结果: {len(train_files)} 训练, {len(valid_files)} 验证, {len(test_files)} 测试。")

            total_files_to_copy = len(train_files) + len(valid_files) + len(test_files)
            if total_files_to_copy == 0:
                self.progress.emit(100)
                self.finished.emit("数据集分割成功，但没有文件需要复制。")
                return

            copied_count = 0

            # 传递正确的源目录给复制函数
            copied_count = self._copy_files(train_files, 'train', copied_count, total_files_to_copy, image_dir,
                                            label_dir)
            copied_count = self._copy_files(valid_files, 'valid', copied_count, total_files_to_copy, image_dir,
                                            label_dir)
            copied_count = self._copy_files(test_files, 'test', copied_count, total_files_to_copy, image_dir, label_dir)

            self.progress.emit(100)
            self.finished.emit("数据集分割成功！")

        except Exception as e:
            import traceback
            self.finished.emit(f"分割失败: {e}\n{traceback.format_exc()}")

    def _random_split(self, basenames):
        total_size = len(basenames)
        train_end = int(total_size * self.config['train_ratio'])
        valid_end = train_end + int(total_size * self.config['valid_ratio'])

        train_files = basenames[:train_end]
        valid_files = basenames[train_end:valid_end]
        test_files = basenames[valid_end:]
        return train_files, valid_files, test_files

    def _stratified_split(self, label_dir, basenames):
        class_to_files = defaultdict(list)
        for name in basenames:
            label_path = os.path.join(label_dir, name + '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    classes_in_file = {line.strip().split()[0] for line in f if line.strip()}
                    if not classes_in_file:  # 处理空标签文件
                        class_to_files['__no_class__'].append(name)
                    for cls in classes_in_file:
                        class_to_files[cls].append(name)
            else:  # 处理没有标签文件的图片
                class_to_files['__no_label_file__'].append(name)

        train_files, valid_files, test_files = set(), set(), set()

        for cls, files in class_to_files.items():
            random.shuffle(files)
            total = len(files)
            train_end = int(total * self.config['train_ratio'])
            valid_end = train_end + int(total * self.config['valid_ratio'])

            train_files.update(files[:train_end])
            valid_files.update(files[train_end:valid_end])
            test_files.update(files[valid_end:])

        valid_files -= train_files
        test_files -= train_files
        test_files -= valid_files

        # 确保所有文件都被分配
        all_assigned = train_files.union(valid_files).union(test_files)
        remaining = list(set(basenames) - all_assigned)
        if remaining:
            self.log.emit(f"警告：分层采样后仍有 {len(remaining)} 个文件未分配，将进行随机分配。")
            train_rem_end = int(len(remaining) * self.config['train_ratio'])
            valid_rem_end = train_rem_end + int(len(remaining) * self.config['valid_ratio'])
            train_files.update(remaining[:train_rem_end])
            valid_files.update(remaining[train_rem_end:valid_rem_end])
            test_files.update(remaining[valid_rem_end:])

        return list(train_files), list(valid_files), list(test_files)

    def _copy_files(self, basenames, split_name, running_count, total_count, source_image_dir, source_label_dir):
        dest_dir = self.config['dest_dir']

        dest_image_dir = os.path.join(dest_dir, split_name, 'images')
        dest_label_dir = os.path.join(dest_dir, split_name, 'labels')
        os.makedirs(dest_image_dir, exist_ok=True)
        os.makedirs(dest_label_dir, exist_ok=True)

        self.log.emit(f"正在复制 {split_name} 文件 ({len(basenames)}个)...")
        for name in basenames:
            source_img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                path = os.path.join(source_image_dir, name + ext)  # 使用传入的源图片目录
                if os.path.exists(path):
                    source_img_path = path
                    break

            if source_img_path:
                shutil.copy2(source_img_path, dest_image_dir)

            source_label_path = os.path.join(source_label_dir, name + '.txt')  # 使用传入的源标签目录
            if os.path.exists(source_label_path):
                shutil.copy2(source_label_path, dest_label_dir)

            running_count += 1
            progress_percent = 10 + int((running_count / total_count) * 90)
            self.progress.emit(progress_percent)

        return running_count