# augmentation_dialog.py
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import Qt
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import Qt

from libs.augmentations_config import AUGMENTATIONS


class AugmentationDialog(QDialog):
    def __init__(self, parent=None):
        super(AugmentationDialog, self).__init__(parent)
        self.setWindowTitle("数据集增强工具")
        self.setMinimumSize(600, 800)
        self.config = None
        self.controls = {}

        # --- UI布局 ---
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        # 源目录和目标目录
        self.source_dir_edit = QLineEdit()
        browse_source_btn = QPushButton("浏览...")
        browse_source_btn.clicked.connect(lambda: self.browse_dir(self.source_dir_edit, "选择源数据集目录"))
        source_layout = QHBoxLayout()
        source_layout.addWidget(self.source_dir_edit)
        source_layout.addWidget(browse_source_btn)
        form_layout.addRow("源目录:", source_layout)

        self.output_dir_edit = QLineEdit()
        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(lambda: self.browse_dir(self.output_dir_edit, "选择输出目录"))
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(browse_output_btn)
        form_layout.addRow("输出目录:", output_layout)

        self.num_augs_spin = QSpinBox()
        self.num_augs_spin.setRange(1, 50)
        self.num_augs_spin.setValue(4)
        form_layout.addRow("每个图像的增强数量:", self.num_augs_spin)

        main_layout.addLayout(form_layout)

        # 增强选项的滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for category, augs in AUGMENTATIONS.items():
            cat_group = QGroupBox(category)
            cat_layout = QFormLayout()
            for aug_name, config in augs.items():
                cb = QCheckBox(aug_name)
                cat_layout.addRow(cb)
                self.controls[aug_name] = {'checkbox': cb, 'config': config, 'params': {}}

                # 动态创建参数控件 (这里简化为仅概率滑块，您可以扩展)
                if 'p' in config['params']:
                    p_config = config['params']['p']
                    slider = QSlider(Qt.Horizontal)
                    slider.setRange(0, 100)
                    slider.setValue(int(p_config['default'] * 100))
                    val_label = QLabel(f"{p_config['default']:.2f}")
                    slider.valueChanged.connect(lambda v, lbl=val_label: lbl.setText(f"{v / 100.0:.2f}"))

                    h_layout = QHBoxLayout()
                    h_layout.addWidget(QLabel(p_config['label']))
                    h_layout.addWidget(slider)
                    h_layout.addWidget(val_label)
                    cat_layout.addRow(h_layout)
                    self.controls[aug_name]['params']['p'] = slider

            cat_group.setLayout(cat_layout)
            scroll_layout.addWidget(cat_group)

        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        # 控制按钮
        self.start_btn = QPushButton("开始增强")
        self.start_btn.clicked.connect(self.accept)
        main_layout.addWidget(self.start_btn)

    def browse_dir(self, line_edit, title):
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def accept(self):
        # 收集配置信息
        self.config = {
            "source_dir": self.source_dir_edit.text(),
            "output_dir": self.output_dir_edit.text(),
            "num_augs": self.num_augs_spin.value(),
            "augmentations": {}
        }
        for aug_name, ctrl in self.controls.items():
            if ctrl['checkbox'].isChecked():
                params_values = {}
                for param_name, widget in ctrl['params'].items():
                    if isinstance(widget, QSlider):
                        params_values[param_name] = widget.value() / 100.0
                self.config['augmentations'][aug_name] = {"class": ctrl['config']['class'], "values": params_values}

        super().accept()