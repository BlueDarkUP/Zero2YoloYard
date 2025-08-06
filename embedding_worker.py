# embedding_worker.py
import numpy as np
from PySide6.QtCore import QObject, Signal

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.manifold import TSNE
    from umap import UMAP
    from PIL import Image

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class EmbeddingWorker(QObject):
    finished = Signal(dict)
    log = Signal(str)

    def __init__(self, all_image_annotations, method='umap'):
        super().__init__()
        if not ML_AVAILABLE:
            raise ImportError("创建EmbeddingWorker需要安装Pytorch, scikit-learn, umap-learn和sentence-transformers。")

        self.all_image_annotations = all_image_annotations
        self.method = method

    def run(self):
        try:
            self.log.emit("嵌入空间分析开始...")

            self.log.emit("加载预训练模型 (CLIP)...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('clip-ViT-B-32', device=device)
            self.log.emit(f"模型已加载到 {device}。")

            self.log.emit("正在提取图像特征...")
            image_paths = [ann.image_path for ann in self.all_image_annotations]
            images_to_encode = [Image.open(p) for p in image_paths]
            embeddings = model.encode(images_to_encode, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
            self.log.emit("特征提取完成。")

            self.log.emit(f"正在使用 {self.method.upper()} 进行降维...")
            if self.method == 'tsne':
                reducer = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
            else:
                reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

            reduced_embeddings = reducer.fit_transform(embeddings)
            self.log.emit("降维完成。")

            point_labels = [ann.annotations[0].class_id if ann.annotations else -1 for ann in
                            self.all_image_annotations]

            results = {
                "embeddings_2d": reduced_embeddings,
                "point_labels": np.array(point_labels)
            }
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.log.emit(f"嵌入空间分析失败: {e}\n{traceback.format_exc()}")
            self.finished.emit(None)