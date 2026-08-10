#!/usr/bin/env python3
"""Minimal batch inference CLI for the LocateAnything-3B release code.

Examples:
  python batch_infer.py --model /path/to/LocateAnything-3B --attn sdpa \
    --image demo.jpg --query "person</c>car"

  python batch_infer.py --requests requests.jsonl --batch-size 16 --attn la_flash

Each JSONL request should contain {"image": "/path/to.jpg", "query": "person</c>car"}.
"""
import argparse
import json
import os
from pathlib import Path

from PIL import Image


def _attn_arg(value):
    mode = (value or "sdpa").strip().lower().replace("-", "_")
    aliases = {
        "": "sdpa",
        "manual": "eager",
        "torch": "eager",
        "torch_eager": "eager",
        "torch_sdpa": "sdpa",
        "flash": "la_flash",
        "la_flash": "la_flash",
        "kernel": "la_flash",
        "cuda": "la_flash",
        "range": "la_flash",
        "range_attention": "la_flash",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"sdpa", "eager", "magi", "la_flash"}:
        raise argparse.ArgumentTypeError(
            f"--attn must be one of sdpa, eager, magi, la_flash; got {value!r}"
        )
    return mode


def _load_requests(args):
    requests = []
    if args.requests:
        with open(args.requests, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                requests.append((row["image"], row["query"]))
    if args.image or args.query:
        if len(args.image or []) != len(args.query or []):
            raise ValueError("--image and --query must appear the same number of times")
        requests.extend(zip(args.image, args.query))
    if not requests:
        raise ValueError("provide --requests JSONL or at least one --image/--query pair")
    return requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", help="JSONL file with image/query fields")
    ap.add_argument("--image", action="append", help="Image path; repeat with --query")
    ap.add_argument("--query", action="append", help="Category query, e.g. person</c>car")
    ap.add_argument("--model", default=os.environ.get("LA_FLASH_MODEL", "nvidia/LocateAnything-3B"))
    ap.add_argument("--attn", type=_attn_arg, default=os.environ.get("LA_FLASH_ATTN", "sdpa"),
                    help="LLM attention backend: sdpa, eager, magi, or la_flash")
    ap.add_argument("--vision-attn", default=os.environ.get("LA_FLASH_VISION_ATTN", "auto"),
                    choices=["auto", "flash_attention_2", "sdpa", "eager"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--scheduler", default=os.environ.get("LA_FLASH_HYBRID_SCHEDULER", "eager"),
                    choices=["eager", "hold_ar", "ar_first", "pipeline", "adaptive"])
    ap.add_argument("--group-size", type=int, default=int(os.environ.get("LA_FLASH_HYBRID_GROUP_SIZE", "0")))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--strict-attn", action="store_true",
                    help="Fail instead of falling back to SDPA if magi/la_flash is unavailable")
    ap.add_argument("--out", default="", help="Optional output JSONL path; stdout if omitted")
    args = ap.parse_args()
    args.attn = _attn_arg(args.attn)

    os.environ["LA_FLASH_MODEL"] = args.model
    os.environ["LA_FLASH_ATTN"] = args.attn
    os.environ["LA_FLASH_VISION_ATTN"] = args.vision_attn
    os.environ["LA_FLASH_HYBRID_SCHEDULER"] = args.scheduler
    os.environ["LA_FLASH_HYBRID_GROUP_SIZE"] = str(args.group_size)
    if args.strict_attn:
        os.environ["LA_FLASH_STRICT_ATTN"] = "1"

    from batch_utils import generate_batch_hybrid, get_last_hybrid_stats, load
    from batch_utils.hybrid_runtime import load_pil

    requests = _load_requests(args)
    load()

    writer = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for start in range(0, len(requests), max(1, args.batch_size)):
            chunk = requests[start:start + max(1, args.batch_size)]
            pairs = [(load_pil(image), query) for image, query in chunk]
            texts = generate_batch_hybrid(
                pairs,
                temperature=args.temperature,
                top_p=None if args.top_p < 0 else args.top_p,
                top_k=None if args.top_k <= 0 else args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                scheduler=args.scheduler,
                group_size=args.group_size,
            )
            stats = get_last_hybrid_stats()
            for (image, query), text in zip(chunk, texts):
                row = {"image": str(Path(image)), "query": query, "raw_response": text, "stats": stats}
                line = json.dumps(row, ensure_ascii=False)
                if writer:
                    writer.write(line + "\n")
                else:
                    print(line, flush=True)
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
