# Batch Utils

`batch_utils` contains the optional batched hybrid generation runtime for
LocateAnything. It keeps the model loading, tokenization, image feature caching,
sampling, and scheduler code used by `batch_infer.py` and the detection
experiments.

## Runtime Modes

- `LA_FLASH_ATTN=sdpa`: stock PyTorch SDPA path.
- `LA_FLASH_ATTN=eager`: eager attention path for debugging.
- `LA_FLASH_ATTN=magi`: MagiAttention path when MagiAttention is installed.
- `LA_FLASH_ATTN=la_flash`: LA Flash sparse range backend
  from `kernel_utils`.

## Common Knobs

| Variable | Default | Meaning |
| --- | --- | --- |
| `LA_FLASH_MODEL` | `nvidia/LocateAnything-3B` | HF model id or local model directory. |
| `LA_FLASH_ATTN` | `sdpa` | LLM attention backend. |
| `LA_FLASH_VISION_ATTN` | `auto` | Vision encoder attention: `auto`, `flash_attention_2`, `sdpa`, or `eager`. |
| `LA_FLASH_STRICT_ATTN` | `0` | Set `1` to fail instead of falling back to SDPA. |
| `LA_FLASH_HYBRID_SCHEDULER` | `eager` | Hybrid decode scheduler. |
| `LA_FLASH_HYBRID_GROUP_SIZE` | `0` | Scheduler group size; `0` lets the runtime decide. |
| `LA_FLASH_VISION_ENCODE_BATCH_SIZE` | `8` | Maximum images per MoonViT encode micro-batch. |
| `LA_FLASH_KV_PACK_TOKEN_BUDGET` | `0` | Optional KV packing memory cap for long-tail batches. |
| `LA_FLASH_DENSE_BACKEND` | `sdpa` | Dense worker/prefill attention backend. Keep this as `sdpa`; LA Flash is used for sparse range plans. |
| `LA_FLASH_SEGMENT_FASTPATH` | `auto` | Sparse MTP decode uses FlashAttention varlen multi-segment merge by default. |

## CLI Example

```bash
python batch_infer.py \
  --model nvidia/LocateAnything-3B \
  --attn la_flash \
  --scheduler pipeline \
  --batch-size 4 \
  --image /path/to/image.jpg \
  --query "person</c>car"
```

For JSONL input, each row should contain:

```json
{"image": "/path/to/image.jpg", "query": "person</c>car"}
```

## Training Boundary

This package is for inference and evaluation. Training remains on the
MagiAttention backend; the batched sparse-plan decode runtime does not support
the `labels` training path.
