from PySide6.QtWidgets import (QDialog, QVBoxLayout, QListWidget, QHBoxLayout,
                               QPushButton, QInputDialog, QMessageBox, QListWidgetItem)
from PySide6.QtCore import Qt


class ClassManagerDialog(QDialog):
    def __init__(self, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("类别管理")
        self.setMinimumSize(400, 500)

        self.class_names = class_names.copy()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        self.class_list = QListWidget()
        self.class_list.addItems(self.class_names)
        layout.addWidget(self.class_list)

        btn_layout = QHBoxLayout()

        self.add_btn = QPushButton("添加")
        self.add_btn.clicked.connect(self.add_class)

        self.rename_btn = QPushButton("重命名")
        self.rename_btn.clicked.connect(self.rename_class)

        self.delete_btn = QPushButton("删除")
        self.delete_btn.clicked.connect(self.delete_class)

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.rename_btn)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

        btn_box = QHBoxLayout()

        self.ok_btn = QPushButton("确定")
        self.ok_btn.clicked.connect(self.accept)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.ok_btn)
        btn_box.addWidget(self.cancel_btn)

        layout.addLayout(btn_box)

    def add_class(self):
        text, ok = QInputDialog.getText(self, '添加类别', '输入新类别名称:')
        if ok and text:
            if text in self.class_names:
                QMessageBox.warning(self, '错误', f'类别 "{text}" 已存在！')
                return
            self.class_names.append(text)
            self.class_list.addItem(text)

    def rename_class(self):
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, '错误', '请先选择一个类别！')
            return

        old_name = current_item.text()
        new_name, ok = QInputDialog.getText(self, '重命名类别', '输入新名称:', text=old_name)

        if ok and new_name and new_name != old_name:
            if new_name in self.class_names:
                QMessageBox.warning(self, '错误', f'类别 "{new_name}" 已存在！')
                return

            index = self.class_names.index(old_name)
            self.class_names[index] = new_name
            current_item.setText(new_name)

    def delete_class(self):
        current_item = self.class_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, '错误', '请先选择一个类别！')
            return

        name = current_item.text()
        reply = QMessageBox.question(
            self, '确认删除',
            f'确定要删除类别 "{name}" 吗？\n\n注意：这将不会删除已标注的框，但会将其类别ID设置为0。',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            index = self.class_list.row(current_item)
            self.class_list.takeItem(index)
            self.class_names.pop(index)

    @staticmethod
    def manage_classes(class_names, parent=None):
        dialog = ClassManagerDialog(class_names, parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.class_names
        return None