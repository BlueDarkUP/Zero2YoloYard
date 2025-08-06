# split_dialog.py
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                               QPushButton, QFileDialog, QSlider, QLabel, QCheckBox,
                               QHBoxLayout, QMessageBox)
from PySide6.QtCore import Qt


class SplitDialog(QDialog):
    """一个用于配置和启动数据集分割的对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("智能数据集分割工具")
        self.setMinimumWidth(500)

        self.config = None

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 源目录
        self.source_dir_edit = QLineEdit()
        browse_source_btn = QPushButton("浏览...")
        browse_source_btn.clicked.connect(self.browse_source)
        source_layout = QHBoxLayout()
        source_layout.addWidget(self.source_dir_edit)
        source_layout.addWidget(browse_source_btn)
        form_layout.addRow("源数据文件夹:", source_layout)

        # 目标目录
        self.dest_dir_edit = QLineEdit()
        browse_dest_btn = QPushButton("浏览...")
        browse_dest_btn.clicked.connect(self.browse_dest)
        dest_layout = QHBoxLayout()
        dest_layout.addWidget(self.dest_dir_edit)
        dest_layout.addWidget(browse_dest_btn)
        form_layout.addRow("目标输出文件夹:", dest_layout)

        # 分割比例
        self.train_label = QLabel("80%")
        self.valid_label = QLabel("10%")
        self.test_label = QLabel("10%")

        self.train_slider = QSlider(Qt.Horizontal)
        self.train_slider.setRange(0, 100)
        self.train_slider.setValue(80)
        self.train_slider.valueChanged.connect(self.update_ratios)

        self.valid_slider = QSlider(Qt.Horizontal)
        self.valid_slider.setRange(0, 100)
        self.valid_slider.setValue(10)
        self.valid_slider.valueChanged.connect(self.update_ratios)

        ratio_layout = QHBoxLayout()
        ratio_layout.addWidget(QLabel("Train:"))
        ratio_layout.addWidget(self.train_slider)
        ratio_layout.addWidget(self.train_label)
        ratio_layout.addWidget(QLabel("Valid:"))
        ratio_layout.addWidget(self.valid_slider)
        ratio_layout.addWidget(self.valid_label)
        ratio_layout.addWidget(QLabel("Test:"))
        ratio_layout.addWidget(self.test_label)
        form_layout.addRow("分割比例:", ratio_layout)

        # 高级选项
        self.stratified_cb = QCheckBox("使用分层采样 (保持类别分布)")
        self.stratified_cb.setChecked(True)
        form_layout.addRow(self.stratified_cb)

        layout.addLayout(form_layout)

        # 按钮
        self.ok_button = QPushButton("开始分割")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

    def browse_source(self):
        path = QFileDialog.getExistingDirectory(self, "选择源数据文件夹")
        if path: self.source_dir_edit.setText(path)

    def browse_dest(self):
        path = QFileDialog.getExistingDirectory(self, "选择目标输出文件夹")
        if path: self.dest_dir_edit.setText(path)

    def update_ratios(self):
        """当滑块移动时，动态更新并规范化比例。"""
        train_val = self.train_slider.value()
        valid_val = self.valid_slider.value()

        if train_val + valid_val > 100:
            valid_val = 100 - train_val
            self.valid_slider.setValue(valid_val)

        test_val = 100 - train_val - valid_val

        self.train_label.setText(f"{train_val}%")
        self.valid_label.setText(f"{valid_val}%")
        self.test_label.setText(f"{test_val}%")

    def accept(self):
        """当点击OK时，验证输入并收集配置。"""
        source = self.source_dir_edit.text()
        dest = self.dest_dir_edit.text()
        if not source or not dest:
            QMessageBox.warning(self, "输入错误", "源文件夹和目标文件夹都不能为空！")
            return
        if source == dest:
            QMessageBox.warning(self, "输入错误", "源文件夹和目标文件夹不能相同！")
            return

        self.config = {
            "source_dir": source,
            "dest_dir": dest,
            "train_ratio": self.train_slider.value() / 100.0,
            "valid_ratio": self.valid_slider.value() / 100.0,
            "stratified": self.stratified_cb.isChecked()
        }
        super().accept()