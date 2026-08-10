"""Minimal local LocateAnything worker used by the GKD detector adapter."""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class LocateAnythingWorker:
    """Load LocateAnything once and run its standard category-set detection task."""

    def __init__(self, model_path, device='cuda', dtype=torch.bfloat16):
        self.device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        self.dtype = dtype if self.device == 'cuda' else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode='hybrid',
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        repetition_penalty=1.1,
        verbose=True,
    ):
        messages = [{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': question},
            ],
        }]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors='pt'
        ).to(self.device)

        response = self.model.generate(
            pixel_values=inputs['pixel_values'].to(self.dtype),
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_grid_hws=inputs.get('image_grid_hws'),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=None if top_k <= 0 else top_k,
            repetition_penalty=repetition_penalty,
            verbose=verbose,
        )
        result = {'answer': response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result['history'] = response[1]
            result['stats'] = response[2]
        return result

    def detect(self, image: Image.Image, categories: list[str], **kwargs):
        """Locate all instances in the supplied category set."""
        category_set = '</c>'.join(categories)
        prompt = (
            'Locate all the instances that matches the following description: '
            f'{category_set}.'
        )
        return self.predict(image, prompt, **kwargs)
