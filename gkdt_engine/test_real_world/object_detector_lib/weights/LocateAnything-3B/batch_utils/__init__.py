"""Hybrid batched inference helpers for NVIDIA LocateAnything-3B."""
from .hybrid_runtime import load, load_pil
from .engine_hybrid import generate_batch_hybrid, generate_batch_grouped_hybrid, get_last_hybrid_stats

__all__ = [
    "load",
    "load_pil",
    "generate_batch_hybrid",
    "generate_batch_grouped_hybrid",
    "get_last_hybrid_stats",
]
__version__ = "0.1.0"
