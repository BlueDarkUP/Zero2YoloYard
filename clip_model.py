import os
import torch
import numpy as np
from PIL import Image
import cv2

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class CLIPModelManager:
    """
    Manager for CLIP (Contrastive Language-Image Pretraining) models.
    Supports local checkpoints stored in checkpoints/clip/ directory.
    """
    def __init__(self, base_checkpoints_dir="checkpoints/clip"):
        self.base_checkpoints_dir = os.path.abspath(base_checkpoints_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loaded_models = {}
        self.loaded_processors = {}
        self.active_model_name = None

    def get_available_models(self):
        """
        Scan checkpoints/clip/ directory for available local model subfolders.
        """
        models = []
        if os.path.exists(self.base_checkpoints_dir):
            for item in os.listdir(self.base_checkpoints_dir):
                full_path = os.path.join(self.base_checkpoints_dir, item)
                if os.path.isdir(full_path):
                    if os.path.exists(os.path.join(full_path, "config.json")):
                        models.append(item)
        
        preferred_order = ["clip-vit-base-patch32", "clip-vit-base-patch16", "clip-vit-large-patch14"]
        sorted_models = [m for m in preferred_order if m in models]
        for m in models:
            if m not in sorted_models:
                sorted_models.append(m)
        return sorted_models

    def load_model(self, model_name=None):
        """
        Load specified CLIP model and processor onto device.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers library is not installed in current environment.")

        available = self.get_available_models()
        if not available:
            raise FileNotFoundError(f"No valid CLIP models found in {self.base_checkpoints_dir}")

        if not model_name or model_name not in available:
            model_name = available[0]

        if model_name in self.loaded_models:
            self.active_model_name = model_name
            return self.loaded_models[model_name], self.loaded_processors[model_name]

        model_path = os.path.join(self.base_checkpoints_dir, model_name)
        print(f"[CLIPManager] Loading model '{model_name}' from {model_path} onto {self.device}...")

        processor = CLIPProcessor.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path).to(self.device)
        model.eval()

        self.loaded_models[model_name] = model
        self.loaded_processors[model_name] = processor
        self.active_model_name = model_name

        print(f"[CLIPManager] Successfully loaded '{model_name}'.")
        return model, processor

    def extract_image_feature_vector(self, image, model_name=None):
        """
        Extract normalized L2 feature vector (512 or 768 dim) for a single image.
        image: PIL.Image or OpenCV BGR numpy array
        """
        model, processor = self.load_model(model_name)

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError("Input image must be OpenCV numpy array or PIL Image.")

        inputs = processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = model.get_image_features(**inputs)
            if hasattr(out, 'image_embeds'):
                image_features = out.image_embeds
            elif hasattr(out, 'pooler_output'):
                image_features = out.pooler_output
            elif isinstance(out, torch.Tensor):
                image_features = out
            else:
                image_features = torch.tensor(out)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        vec = image_features.squeeze(0).cpu().numpy()
        return vec

    def predict_zero_shot(self, image, candidate_classes, prompt_template="a photo of a {}", use_ensemble=True, model_name=None):
        """
        Industry-standard Zero-Shot Classification using Prompt Ensembling (FiftyOne / OpenAI best practices).
        Supports multi-template prompt averaging & negative background class filtering.
        """
        if not candidate_classes:
            return []

        model, processor = self.load_model(model_name)

        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError("Input image must be OpenCV numpy array or PIL Image.")

        # 1. Image feature extraction
        img_inputs = processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_out = model.get_image_features(**img_inputs)
            if hasattr(img_out, 'image_embeds'):
                img_feat = img_out.image_embeds
            elif hasattr(img_out, 'pooler_output'):
                img_feat = img_out.pooler_output
            elif isinstance(img_out, torch.Tensor):
                img_feat = img_out
            else:
                img_feat = torch.tensor(img_out)
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)

        # 2. Candidate classes preparation + optional background negative class
        import re
        effective_classes = list(candidate_classes)
        has_negative_class = False
        if len(effective_classes) == 1:
            effective_classes.append("other background, floor or irrelevant object")
            has_negative_class = True

        # 3. Prompt Ensembling templates
        if use_ensemble:
            templates = [
                "a photo of a {}.",
                "a close-up photo of the {}.",
                "a cropped photo of a {}.",
                "a bright photo of the {}.",
                "a good photo of a {}.",
                "a photo of the {}.",
                "a small photo of a {}.",
                "a picture of the {}."
            ]
            if prompt_template and prompt_template != "a photo of a {}":
                templates.insert(0, prompt_template)
        else:
            templates = [prompt_template or "a photo of a {}"]

        # 4. Text embeddings computation with Prompt Averaging per class
        class_vectors = []
        with torch.no_grad():
            for c in effective_classes:
                c_prompts = []
                for tmpl in templates:
                    if re.search(r'\{.*?\}', tmpl):
                        c_prompts.append(re.sub(r'\{.*?\}', c, tmpl))
                    else:
                        c_prompts.append(f"{tmpl} {c}")

                txt_inputs = processor(text=c_prompts, return_tensors="pt", padding=True).to(self.device)
                txt_out = model.get_text_features(**txt_inputs)
                if hasattr(txt_out, 'text_embeds'):
                    txt_feat = txt_out.text_embeds
                elif hasattr(txt_out, 'pooler_output'):
                    txt_feat = txt_out.pooler_output
                elif isinstance(txt_out, torch.Tensor):
                    txt_feat = txt_out
                else:
                    txt_feat = torch.tensor(txt_out)
                
                txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
                avg_vec = txt_feat.mean(dim=0, keepdim=True)
                avg_vec = avg_vec / avg_vec.norm(p=2, dim=-1, keepdim=True)
                class_vectors.append(avg_vec)

        class_matrix = torch.cat(class_vectors, dim=0) # [num_classes, embed_dim]

        # 5. Cosine similarity & Softmax with model logit_scale
        with torch.no_grad():
            logit_scale = model.logit_scale.exp() if hasattr(model, 'logit_scale') else 100.0
            logits = (img_feat @ class_matrix.T) * logit_scale
            probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

        results = []
        for i, c_name in enumerate(effective_classes):
            if has_negative_class and c_name == "other background, floor or irrelevant object":
                continue
            results.append({
                'class_name': c_name,
                'score': float(probs[i])
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results


clip_manager = CLIPModelManager()
