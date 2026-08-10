"""FlashAttention sparse range utilities for LocateAnything batch inference."""

from .range_attention import range_attention, is_available

__all__ = ["range_attention", "is_available"]
