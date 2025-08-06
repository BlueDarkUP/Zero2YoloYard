import numpy as np
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTabWidget, QWidget,
                               QTextEdit, QSizePolicy, QHBoxLayout, QPushButton,
                               QMessageBox, QLabel, QProgressDialog)
from PySide6.QtCore import Qt, Signal, QThread
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import RectangleSelector

try:
    from embedding_worker import ML_AVAILABLE, EmbeddingWorker
except ImportError:
    ML_AVAILABLE = False


def find_chinese_font():
    font_names = ['SimHei', 'Microsoft YaHei', 'Source Han Sans CN', 'Noto Sans CJK SC']
    for font_name in font_names:
        try:
            if font_manager.findfont(font_manager.FontProperties(family=font_name)):
                return font_name
        except:
            continue
    return None


CHINESE_FONT = find_chinese_font()
if CHINESE_FONT:
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
    print(f"找到并设置中文字体: {CHINESE_FONT}")
else:
    print("警告: 未在系统中找到可用的中文字体，图表中的中文可能无法正常显示。")

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class DashboardDialog(QDialog):
    filter_requested = Signal(list)

    def __init__(self, analysis_results, all_image_annotations, parent=None):
        super().__init__(parent)
        self.results = analysis_results
        self.all_image_annotations = all_image_annotations
        self.setWindowTitle("交互式数据集仪表盘")
        self.setMinimumSize(1000, 750)

        self.embedding_thread = None
        self.embedding_worker = None
        self.rs = None

        self.main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.create_class_distribution_tab()
        self.create_size_distribution_tab()
        self.create_annotation_density_tab()
        self.create_bbox_analysis_tab()
        if ML_AVAILABLE:
            self.create_embedding_space_tab()
        self.create_health_check_tab()

        self.main_layout.addWidget(self.tabs)

    def on_bar_pick(self, event):
        if event.mouseevent.button != 1: return

        class_id = int(event.artist.get_x() + event.artist.get_width() / 2)
        class_name = self.results['class_names'][class_id]

        filtered_indices = []
        for i, ann_obj in enumerate(self.all_image_annotations):
            for ann in ann_obj.annotations:
                if ann.class_id == class_id:
                    filtered_indices.append(i)
                    break

        QMessageBox.information(self, "筛选", f"将筛选显示所有包含 '{class_name}' 的图像。")
        self.filter_requested.emit(filtered_indices)
        self.accept()

    def on_bbox_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        min_w, max_w = min(x1, x2), max(x1, x2)
        min_h, max_h = min(y1, y2), max(y1, y2)

        filtered_indices = set()
        for i, ann_obj in enumerate(self.all_image_annotations):
            for ann in ann_obj.annotations:
                bbox_w_px = ann.bbox.width * ann_obj.width
                bbox_h_px = ann.bbox.height * ann_obj.height
                if min_w <= bbox_w_px <= max_w and min_h <= bbox_h_px <= max_h:
                    filtered_indices.add(i)

        QMessageBox.information(self, "筛选", f"将筛选 {len(filtered_indices)} 张包含指定尺寸边界框的图像。")
        self.filter_requested.emit(list(filtered_indices))
        self.accept()

    def create_class_distribution_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = MatplotlibCanvas(tab)
        ax = canvas.fig.add_subplot(111)

        class_names = self.results['class_names']
        counts = self.results['class_counts']

        ax.bar(range(len(class_names)), counts, color='#558de8', picker=5)
        ax.set_title('类别分布 (点击条形图进行筛选)', fontsize=16)
        ax.set_ylabel('实例数量', fontsize=12)
        ax.set_xlabel('类别名称', fontsize=12)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        canvas.fig.tight_layout()
        canvas.fig.canvas.mpl_connect('pick_event', self.on_bar_pick)

        layout.addWidget(canvas)
        self.tabs.addTab(tab, "📊 类别分布")

    def create_size_distribution_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = MatplotlibCanvas(tab)
        ax = canvas.fig.add_subplot(111)
        widths = self.results['image_widths']
        heights = self.results['image_heights']

        ax.hist(widths, bins=50, alpha=0.7, color='#558de8', label='宽度')
        ax.hist(heights, bins=50, alpha=0.7, color='#e85555', label='高度')
        ax.set_title('图像尺寸分布', fontsize=16)
        ax.set_xlabel('像素尺寸', fontsize=12)
        ax.set_ylabel('图像数量', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')
        canvas.fig.tight_layout()

        layout.addWidget(canvas)
        self.tabs.addTab(tab, "🖼️ 图像尺寸")

    def create_annotation_density_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = MatplotlibCanvas(tab)
        ax = canvas.fig.add_subplot(111)
        counts = self.results['annotations_per_image']
        if not counts: return

        bins_range = range(min(counts), max(counts) + 2)
        ax.hist(counts, bins=bins_range, color='#55e88d', edgecolor='black', align='left')
        ax.set_title('标注密度分析 (每张图像的目标数量)', fontsize=16)
        ax.set_xlabel('每张图像的目标数量', fontsize=12)
        ax.set_ylabel('图像数量', fontsize=12)
        ax.set_xticks(np.arange(min(counts), max(counts) + 1, step=max(1, (max(counts) - min(counts)) // 20)))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')
        canvas.fig.tight_layout()

        layout.addWidget(canvas)
        self.tabs.addTab(tab, " densité 标注密度")

    def create_bbox_analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        canvas = MatplotlibCanvas(tab)
        ax = canvas.fig.add_subplot(111)
        widths = self.results['bbox_widths_px']
        heights = self.results['bbox_heights_px']

        ax.scatter(widths, heights, alpha=0.1, color='#f39c12', edgecolors='none')
        ax.set_title('边界框尺寸分布 (拖动鼠标框选以筛选)', fontsize=16)
        ax.set_xlabel('宽度 (像素)', fontsize=12)
        ax.set_ylabel('高度 (像素)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='x', labelcolor='white')
        ax.tick_params(axis='y', labelcolor='white')
        canvas.fig.tight_layout()

        self.rs = RectangleSelector(ax, self.on_bbox_select, useblit=True, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True,
                                    props=dict(facecolor='red', edgecolor='white', alpha=0.3))

        layout.addWidget(canvas)
        self.tabs.addTab(tab, "框尺寸")

    def create_embedding_space_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        button_layout = QHBoxLayout()
        self.compute_btn = QPushButton("🚀 计算并显示嵌入空间 (可能需要几分钟并下载模型)")
        self.compute_btn.clicked.connect(self.start_embedding_computation)
        button_layout.addWidget(self.compute_btn)
        layout.addLayout(button_layout)

        self.embedding_canvas = MatplotlibCanvas(tab)
        layout.addWidget(self.embedding_canvas)
        self.tabs.addTab(tab, "✨ 嵌入空间")

    def start_embedding_computation(self):
        self.compute_btn.setEnabled(False)
        self.compute_btn.setText("正在计算中... (请查看主窗口日志)")

        self.embedding_thread = QThread()
        self.embedding_worker = EmbeddingWorker(self.all_image_annotations)
        self.embedding_worker.moveToThread(self.embedding_thread)

        self.embedding_thread.started.connect(self.embedding_worker.run)
        self.embedding_worker.finished.connect(self.on_embedding_complete)
        self.embedding_worker.log.connect(self.parent().log_output.append)

        self.embedding_thread.start()

    def on_embedding_complete(self, results):
        self.compute_btn.setText("计算完成！")
        if self.embedding_thread:
            self.embedding_thread.quit()
            self.embedding_thread.wait()

        if not results:
            QMessageBox.critical(self, "错误", "嵌入空间计算失败，请查看主窗口日志。")
            self.compute_btn.setText("计算失败，请重试")
            self.compute_btn.setEnabled(True)
            return

        ax = self.embedding_canvas.fig.add_subplot(111)
        ax.clear()
        points = results['embeddings_2d']
        labels = results['point_labels']

        scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', alpha=0.6, s=10)
        ax.set_title("图像嵌入空间可视化 (UMAP)", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

        class_names = self.results['class_names']
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=class_names[i] if i < len(class_names) else '无标签',
                       markerfacecolor=scatter.cmap(scatter.norm(i if i != -1 else max(labels) + 1)), markersize=8)
            for i in np.unique(labels)]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        self.embedding_canvas.fig.tight_layout()
        self.embedding_canvas.draw()

    def create_health_check_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        report_text.setLineWrapMode(QTextEdit.NoWrap)

        report_content = """<style>h1{color:#558de8;} h2{color:#f0f0f0;border-bottom:1px solid #555;padding-bottom:5px;} p{color:#ccc;} li{color:#bbb;} .ok{color:#55e88d;} .error{color:#e85555;}</style><h1>数据集健康检查报告</h1>"""

        def format_section(title, items):
            if not items:
                return f"<h2><span class='ok'>✅ {title}</span></h2><p>未发现问题。</p><hr>"
            else:
                content = f"<h2><span class='error'>❌ {title} ({len(items)} 个问题)</span></h2><ul>"
                for item in items[:50]: content += f"<li>{item}</li>"
                if len(items) > 50: content += "<li>... 等等 (仅显示前50个)</li>"
                content += "</ul><hr>"
                return content

        report_content += format_section("缺失标签的图像", self.results['images_without_labels'])
        report_content += format_section("空图像 (标签存在但图像丢失)", self.results['labels_without_images'])
        report_content += format_section("无标注的标签文件", self.results['empty_label_files'])
        report_text.setHtml(report_content)

        layout.addWidget(report_text)
        self.tabs.addTab(tab, "🩺 健康检查")