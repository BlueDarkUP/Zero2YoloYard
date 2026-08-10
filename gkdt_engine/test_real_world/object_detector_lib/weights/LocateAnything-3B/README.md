---
license: other
license_name: nvidia-license
license_link: https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE
language:
- en
tags:
- nvidia
- eagle
- vision
- object-detection
- grounding
- locateanything
- arxiv:2605.27365
demo: https://huggingface.co/spaces/nvidia/LocateAnything
github: https://github.com/NVlabs/Eagle/tree/main/Embodied
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen2.5-3B-Instruct
---

# LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding

<p align="center">
  <img src="assets/teaser.jpg" alt="LocateAnything teaser" width="100%">
</p>

## 🔗 Quick Links

* 🚀 **Online Demo**: [LocateAnything (Hugging Face Spaces)](https://huggingface.co/spaces/nvidia/LocateAnything)
* 💻 **GitHub Code**: [NVlabs/Eagle/Embodied](https://github.com/NVlabs/Eagle/tree/main/Embodied)
* 📄 **Paper**: [arXiv:2605.27365](https://arxiv.org/abs/2605.27365)


# Model Overview

### Description:

LocateAnything is a vision-language model for fast and high-quality visual grounding, enabling precise object localization, dense detection, and point-based localization across diverse domains in both Enterprise Intelligence and Physical AI. The model adopts a generalist design, supporting tasks such as referring expression grounding, multi-object detection, GUI element grounding, and text localization, with strong performance in complex and cluttered scenes.

Its core innovation, Parallel Box Decoding (PBD), predicts complete bounding box coordinates in a single parallel step rather than autoregressive token-by-token decoding, improving efficiency while preserving geometric consistency. This enables up to 2.5× higher throughput compared to prior approaches.

The model is trained on a large-scale multi-domain dataset (12M images, 138M+ queries, 785M bounding boxes) spanning natural scenes, robotics, driving, GUI interaction, and document understanding. It serves as a foundation for generalist multimodal perception and has been integrated into NVIDIA’s frontier production-grade vision-language models, such as Nemotron 3 Nano Omni, supporting grounding, GUI understanding, and multimodal agentic capabilities.

LocateAnything is developed as part of the [Eagle VLM](https://github.com/NVlabs/EAGLE) model family. This released model is for research and development only. In addition, LocateAnything contributed to [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) and [Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) as part of the Computer Use and Visual Grounding features. We give special thanks the Nemotron and Cosmos Teams for time and efforts in product integration.

### Demo Videos

<p align="left">
  <video src="https://huggingface.co/nvidia/LocateAnything-3B/resolve/main/assets/demo.mp4" controls="controls" width="80%">
    Your browser does not support the video tag.
  </video>
</p>

<p align="left">
  <video src="https://huggingface.co/nvidia/LocateAnything-3B/resolve/main/assets/decoding_demo.mp4" controls="controls" width="80%">
    Your browser does not support the video tag.
  </video>
</p>

### License/Terms of Use:

This model is released under the [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE) for non-commercial use, which permits use, reproduction, and modification for **academic and non-profit research purposes only**. Commercial use is **not permitted**, except by NVIDIA and its affiliates. Redistribution must retain the license and all applicable copyright and attribution notices. The model is provided **“as is” without warranty of any kind**, and users assume all associated risks.

This model is built using components from third-party models with their respective licenses:
- Language model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Qwen Research License)
- Vision encoder: [MoonViT-SO-400M](https://huggingface.co/moonshotai/MoonViT-SO-400M) (MIT License)

Models are improved using Qwen.

### Deployment Geography:

Global

### Use Case:

LocateAnything-3B is intended for developers and researchers building vision-language models and applications that require fast and precise visual localization from natural language instructions.

Supported use cases include:
- Open-set, common, and long-tail object detection
- Dense multi-object detection in cluttered scenes
- Phrase and referring-expression grounding
- Automated dataset labeling and annotation (e.g., detection, grounding, pointing)
- GUI element grounding for interactive and agentic systems
- Robotics and autonomous driving perception
- Document understanding, layout grounding, and OCR localization
- Industrial inspection, surveillance, and remote sensing applications
- Point-based localization and fine-grained spatial reasoning

### Release Date [Insert the expected release date below]:

- Github [05/26/2026] via https://github.com/NVlabs/Eagle/tree/main/Embodied.
- Hugging Face [05/26/2026] via https://huggingface.co/nvidia/LocateAnything-3B.
- Demo [05/26/2026] via https://huggingface.co/spaces/nvidia/LocateAnything.
- Webpage [05/26/2026] via https://research.nvidia.com/labs/lpr/locate-anything/.
- Tech Report [05/26/2026] via https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf

## References(s):
- Wang et al., [LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf), NVIDIA Tech Report, 2026
- Kimi Team, [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491), arXiv:2504.07491, 2025.
- Qwen Team, [Qwen2.5: A Party of Foundation Models](https://qwen.ai/blog?id=qwen2.5), Qwen Blog, 2024.
- Chen et al., [Pix2Seq: A Language Modeling Framework for Object Detection](https://arxiv.org/abs/2109.10852), ICLR, 2022.
- Jiang et al., [Detect Anything via Next Point Prediction](https://arxiv.org/abs/2510.12798), arXiv:2510.12798, 2025.
- Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), arXiv:2303.05499, 2023.
- Lin et al., [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV, 2014.
- Gupta et al., [LVIS: A Dataset for Large Vocabulary Instance Segmentation](https://arxiv.org/abs/1908.03195), CVPR, 2019.
- Li et al., [ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use](https://arxiv.org/abs/2504.07981), ACM MM, 2025.

## Model Architecture:

**Architecture Type:** Transformer-based vision-language model (VLM).  

**Network Architecture:** Native-resolution VLM with the following components:
- Vision encoder: MoonViT
- Language model: Qwen2.5-3B-Instruct
- Multimodal projector: MLP projector
- Output formulation: Block-based structure for visual grounding

**Number of model parameters:** 3B.

LocateAnything extends a vision-language model with Parallel Box Decoding (PBD), a block-wise multi-token prediction framework for efficient visual grounding. Instead of autoregressive coordinate generation, the model predicts complete bounding boxes and points in parallel structured units, improving decoding efficiency while preserving geometric consistency. The architecture jointly optimizes next-token prediction and multi-token prediction to balance reasoning ability and parallel inference. Training follows a four-stage pipeline: initial multimodal knowledge adaptation using captioning, VQA, OCR, and related data, followed by grounding and dense-scene localization fine-tuning.

## Input(s):

**Input Type(s):** Image and Text.

**Input Format(s):**
- Image: RGB image input with original source resolution.
- Text: Natural-language prompt or task template, such as object categories, referring expressions, GUI instructions, OCR/layout requests, or pointing queries.

**Input Parameters:**
- Image: Two-Dimensional (2D)
- Text: One-Dimensional (1D)

**Other Properties Related to Input:**
- Production image resolution supports up to 2.5K.
- Prompt length supports up to 24K tokens.
- Training detection and grounding stages use a maximum sequence length of 25,600 tokens.
- Inference supports up to 8,192 newly generated tokens.

## Output(s):

**Output Type(s):** Text.

**Output Format(s):**
- Text: Model-generated token sequence containing semantic labels and structured coordinate tokens, such as bounding boxes (`<box> x1, y1, x2, y2 </box>`) and points (`<box> x, y </box>`).

**Output Parameters:**
- Text: One-Dimensional (1D)
- Bounding boxes/points: Two-Dimensional (2D) spatial coordinates

**Other Properties Related to Output:**
- Outputs are organized into fixed-length blocks (length 6), including Semantic, Box, Negative, and End blocks.
- A Box block encodes quantized spatial coordinates with structural tokens; unused positions are padded with `<null>`.
- Fast Mode predicts box-aligned blocks in parallel; Slow Mode uses autoregressive decoding; Hybrid Mode defaults to parallel decoding with fallback to autoregressive decoding for format irregularity or spatial ambiguity.

Our AI models are designed and optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA hardware (e.g., GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves improved training and inference performance compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** 
* Transformers. The inference setup uses standard VLM generation with BF16 precision and KV cache. TensorRT, TensorRT-LLM, and Triton are not yet supported.

**Supported Hardware Microarchitecture Compatibility:**

* NVIDIA Ampere (e.g., A100)
* NVIDIA Blackwell
* NVIDIA Hopper (e.g., H100)
* NVIDIA Lovelace (e.g., L40, RTX 4090)

Deployment on embedded platforms such as NVIDIA Thor is possible with additional model optimization, including quantization, compression, or distillation. Other architectures may be supported depending on available memory, precision support, and software configuration.

**Supported Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
LocateAnything-3B: 3B-parameter research model variant evaluated in Hybrid Mode by default. Fast, Hybrid, and Slow inference modes are supported by the same model formulation.

LocateAnything-3B can be integrated into systems that require spatial grounding from natural language, such as GUI agents, robotics/embodied agents, document-understanding pipelines, OCR/text localization, and open-world detection workflows.

## Training, Testing, and Evaluation Datasets:

### Data Modality:
Image and Text. <br>
* Image <br>
* Text <br>

### Training Data Size: 
**Image Training Data Size:** <br>
* 1 Million to 1 Billion Images - 12M unique images. <br>

**Text Training Data Size:** <br>
* 1 Billion to 10 Trillion Tokens - Derived from approximately 140M natural-language queries. <br>

**Data Collection Method by dataset:** <br>
- Hybrid: Human, Automated <br>
  Data is collected from human-curated and open-source datasets, as well as automated ingestion of publicly available data sources.

**Labeling Method by dataset:** <br>
- Hybrid: Human, Synthetic, Automated <br>
  Labeling includes original human or open-source annotations, along with model-assisted and synthetic annotation generation using Qwen3-VL, Molmo, SAM 3, and Rex-Omni, with automated post-verification.

**Properties:** The training data consists of supervised fine-tuning (SFT) datasets with multimodal inputs, primarily image-text pairs and structured annotations such as bounding boxes, points, and negative samples.

The data spans multiple domains, including grounding, open-world grounding, general and dense object detection, scene text detection, GUI understanding and grounding, document layout understanding, and OCR.

Modalities include visual inputs (images) and natural-language queries or instructions. The dataset is derived from a mixture of publicly available academic datasets, along with model-assisted and synthetic annotations. It may include publicly available and potentially copyrighted content; users are responsible for ensuring compliance with applicable usage rights.

The linguistic content primarily consists of short, task-oriented natural-language expressions, such as object categories, referring expressions, GUI instructions, OCR queries, and grounding prompts, typically in English.

## Evaluation Dataset:

**Data Collection Method by dataset:**
- Hybrid: Human, Automated

**Labeling Method by dataset:**
- Hybrid: Human, Synthetic, Automated

**Properties:** The evaluation datasets consist of publicly available benchmarks spanning visual grounding, object detection, document understanding, scene text detection, and GUI-related tasks. Modalities include image inputs paired with natural-language queries and structured annotations such as bounding boxes and points.

The evaluation suite covers both box-level and point-level grounding tasks, with approximately 48K images for box evaluation and 35K images for point evaluation across multiple datasets. These datasets span diverse domains including natural scenes, documents, aerial imagery, and human-centric interactions, enabling comprehensive assessment of localization accuracy and robustness.

Evaluation queries are typically short, task-oriented natural-language expressions such as referring phrases, object categories, and grounding prompts.

Performance is measured using box-based F1 at IoU thresholds of 0.5 and 0.95, as well as mean IoU for detection, layout, and OCR tasks. Point-based localization is evaluated based on whether predicted points fall within ground-truth segmentation masks or bounding boxes. Inference efficiency is reported in boxes per second (BPS) on a single NVIDIA H100 GPU with batch size 1.

## Quantitative Evaluation Benchmarks

### General Object Detection
<p align="left">
  <img src="assets/coco_lvis.png" width="700">
</p>

### Dense Object Detection
<p align="left">
  <img src="assets/dense_object_detection.png" width="700">
</p>

### GUI Understanding
<p align="left">
  <img src="assets/sspro.png" width="700">
</p>

### Layout Grounding and OCR
<p align="left">
  <img src="assets/layout_ocr.png" width="700">
</p>

### Referring Expression Grounding
<p align="left">
  <img src="assets/referring.png" width="700">
</p>

### Pointing
<p align="left">
  <img src="assets/pointing.png" width="700">
</p>

## Inference:

Test Hardware: H100 & A100

We suggest using `max_new_tokens=8192` and `generation_mode="hybrid"` to avoid truncated response and balance speed with accuracy.

### Batch Hybrid Inference

This release includes `batch_infer.py`, `batch_utils`, and `kernel_utils` for
high-throughput detection and grounding. The `la_flash` backend is a pure
FlashAttention-varlen sparse range executor: it keeps LocateAnything's hybrid
MTP decoding path, avoids dense `[B,H,Q,K]` SDPA masks, and does not require a
custom CUDA extension build.

Use it with:

```bash
python batch_infer.py \
  --model . \
  --attn la_flash \
  --scheduler pipeline \
  --batch-size 4 \
  --image /path/to/image.jpg \
  --query "person</c>car"
```

A100 4K probe, real 3840x2160 street image, `query=vehicle`,
`batch_size=4`, raw PIL input, `in_token_limit=25600`, hybrid MTP inference:

| Backend | Attention Path | Time | Peak Reserved Memory |
| --- | --- | ---: | ---: |
| `sdpa` | Dense SDPA masks | 8.2600 s | 35.12 GB |
| `la_flash` | FlashAttention sparse range plan | 8.0314 s | 11.71 GB |

See `batch_utils/README.md` and `kernel_utils/README.md` for runtime knobs and
implementation details.

### Installation

```bash
pip install opencv-python-headless==4.11.0.86 transformers==4.57.1 numpy==1.25.0 Pillow==11.1.0 peft torchvision decord==0.6.0 lmdb==1.7.5
```

> PyTorch (`torch`) must be installed separately according to your CUDA version. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

Optional — [MagiAttention](https://sandai-org.github.io/MagiAttention/docs/main/user_guide/install.html) (Hopper / Blackwell GPUs only, recommended for faster MTP inference):

```bash
git clone https://github.com/SandAI-org/MagiAttention.git
cd MagiAttention
git checkout v1.0.5
git submodule update --init --recursive
pip install -r requirements.txt
pip install --no-build-isolation .
```

If MagiAttention is installed, the model will automatically use it for efficient MTP block-diffusion attention. If not installed, it will fall back to PyTorch SDPA — fully functional but slower for MTP decoding.

### Worker (recommended)

Below is a self-contained worker that loads the model once and serves perception queries via a unified `predict()` plus task-specific convenience methods. You can drop this class into any FastAPI / gRPC / Triton serving framework.

```python
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (NTP/AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points
```

### Usage Example

```python
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
img = Image.open("example.jpg").convert("RGB")

# Object Detection
result = worker.detect(img, ["person", "car", "bicycle"])
print("Detection:", result["answer"])

# Phrase Grounding (multiple)
result = worker.ground_multi(img, "people wearing red shirts")
print("Grounding:", result["answer"])

# Scene Text Detection
result = worker.detect_text(img)
print("Text Detection:", result["answer"])

# Pointing
result = worker.point(img, "the traffic light")
print("Pointing:", result["answer"])

# GUI Grounding (point)
result = worker.ground_gui(img, "the search button", output_type="point")
print("GUI Point:", result["answer"])

# Parse structured output into pixel coordinates
w, h = img.size
boxes = LocateAnythingWorker.parse_boxes(result["answer"], w, h)
points = LocateAnythingWorker.parse_points(result["answer"], w, h)
```

### Supported Tasks & Prompt Templates

| Task | Worker Method | Output | Prompt Template |
| --- | --- | --- | --- |
| Object Detection | `worker.detect(img, [...])` | Box | `Locate all the instances that matches the following description: [CATEGORIES].` |
| Phrase Grounding (single) | `worker.ground_single(img, phrase)` | Single Box | `Locate a single instance that matches the following description: [PHRASE].` |
| Phrase Grounding (multi) | `worker.ground_multi(img, phrase)` | Multiple Boxes | `Locate all the instances that match the following description: [PHRASE].` |
| Text Grounding | `worker.ground_text(img, phrase)` | Box | `Please locate the text referred as [PHRASE].` |
| Scene Text Detection | `worker.detect_text(img)` | Box | `Detect all the text in box format.` |
| Document Layout Analysis | `worker.detect(img, [...])` | Box | `Locate all the instances that matches the following description: [CATEGORIES].` |
| GUI Grounding (box) | `worker.ground_gui(img, phrase, "box")` | Box | `Locate the region that matches the following description: [PHRASE].` |
| GUI Grounding (point) / Pointing | `worker.ground_gui(img, phrase, "point")` / `worker.point(img, phrase)` | Point | `Point to: [PHRASE].` |

`[PHRASE]` is a free-form natural-language description; `[CATEGORIES]` is a comma-separated list (multiple categories may also be joined with `</c>`).

### Generation Modes

| Mode | Description | Speed | Accuracy |
| --- | --- | --- | --- |
| `fast` | MTP only, never falls back to AR | Fastest | Good for simple scenes |
| `slow` | Pure auto-regressive decoding | Slowest | Most robust |
| `hybrid` (default) | MTP first, falls back to AR on uncertain boxes, switches back after box boundary | Balanced | Best overall |

## Batch Utils and Kernel Utils

This repository also includes optional utilities for high-throughput detection
runs:

- `batch_infer.py`: JSONL/image-query batch inference CLI.
- `batch_utils/`: batched hybrid generation runtime. See
  `batch_utils/README.md`.
- `kernel_utils/`: LA Flash sparse range utilities. See
  `kernel_utils/README.md`.

Run a small batch inference job:

```bash
python batch_infer.py \
  --model . \
  --attn la_flash \
  --scheduler pipeline \
  --batch-size 4 \
  --image assets/pointing.png \
  --query "the object being pointed at"
```

The batched sparse-plan decode runtime is intended for inference/evaluation and
does not support the training `labels` path. Training remains on the
MagiAttention backend.

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
