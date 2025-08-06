# ai_models.py
import os
import torch
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor
    import wget

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("警告: 缺少 AI 依赖库 (ultralytics, segment-anything, torch, wget)。AI 辅助功能将被禁用。")
    print("请运行: pip install ultralytics torch torchvision segment-anything wget")


class ModelManager:
    def __init__(self, model_name='sam_vit_b'):
        if not AI_AVAILABLE:
            raise ImportError("AI 依赖库未安装。")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.predictor = None
        self.model_name = model_name
        self._load_model()

    def _get_model_path(self):
        home = os.path.expanduser("~")
        model_dir = os.path.join(home, ".labelImg_models")
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f"{self.model_name}.pth")

    def _load_model(self):
        model_path = self._get_model_path()

        model_urls = {
            'sam_vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'sam_vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'sam_vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        }

        if not os.path.exists(model_path):
            print(f"正在下载 SAM 模型 '{self.model_name}'...")
            wget.download(model_urls[self.model_name], model_path)
            print("\n下载完成。")

        self.model = sam_model_registry[self.model_name.split('_')[1]](checkpoint=model_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def run_sam_inference(self, image_np, point_coords, point_labels):
        if self.predictor is None:
            raise RuntimeError("SAM predictor 未初始化。")

        self.predictor.set_image(image_np)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,  # 我们只想要最佳掩码
        )

        return masks[0]  # 返回单个掩码