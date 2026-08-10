"""Sparse LocateAnything attention implemented with FlashAttention varlen.

The public API accepts flattened query/key/value tensors:

  q: [total_q, num_q_heads, head_dim]
  k: [total_k, num_kv_heads, head_dim]
  v: [total_k, num_kv_heads, head_dim]

and a Magi-style range plan:

  q_ranges: [num_ranges, 2]
  k_ranges: [num_key_segments, 2]
  segment_offsets: [num_query_groups + 1]
  attn_type_map:
    0 = full attention over the listed key segment(s)
    1 = bottom-right causal attention

For LocateAnything hybrid MTP decode, batch_utils represents the window as a
causal prefix plus full-attention sparse window segments.  This module packs
those visible KV segments and calls FlashAttention varlen, avoiding dense masks.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


_FLASH_ATTN_VARLEN = None
_FLASH_ATTN_ERROR: Optional[BaseException] = None


def _env_enabled(name: str, default: str = "auto") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"", "auto", "1", "on", "true", "yes", "force"}


def is_available() -> bool:
    try:
        _load_flash_attn_varlen()
        return True
    except Exception:
        return False


def _flash_fastpath_enabled() -> bool:
    return _env_enabled("LA_FLASH_FASTPATH", "auto")


def _flash_segment_fastpath_enabled() -> bool:
    return _env_enabled("LA_FLASH_SEGMENT_FASTPATH", "auto")


def _load_flash_attn_varlen():
    global _FLASH_ATTN_VARLEN, _FLASH_ATTN_ERROR
    if _FLASH_ATTN_VARLEN is not None:
        return _FLASH_ATTN_VARLEN
    if _FLASH_ATTN_ERROR is not None:
        raise _FLASH_ATTN_ERROR
    try:
        from flash_attn import flash_attn_varlen_func

        _FLASH_ATTN_VARLEN = flash_attn_varlen_func
        return _FLASH_ATTN_VARLEN
    except BaseException as exc:
        _FLASH_ATTN_ERROR = exc
        raise


def _coalesce_query_groups(q_ranges, k_ranges, attn_type_map):
    """Group consecutive entries that share the same query span and mask type."""
    if q_ranges.numel() == 0:
        segment_offsets = torch.zeros((1,), dtype=torch.int32, device=q_ranges.device)
        return q_ranges, k_ranges, segment_offsets, attn_type_map, 0, 0

    q_cpu = q_ranges.detach().to(device="cpu", dtype=torch.int32).contiguous()
    t_cpu = attn_type_map.detach().to(device="cpu", dtype=torch.int32).contiguous()
    grouped_q = []
    grouped_t = []
    offsets = [0]
    max_q_len = 0
    last_q = None
    last_t = None
    for idx, (qr, attn_type) in enumerate(zip(q_cpu.tolist(), t_cpu.tolist())):
        key = (int(qr[0]), int(qr[1]))
        attn_type = int(attn_type)
        if attn_type not in (0, 1):
            raise RuntimeError(
                "LA Flash path only supports FlashAttention-compatible attn_type 0/1. "
                f"Got attn_type={attn_type}; regenerate a type 0/1 range plan."
            )
        if last_q is None:
            grouped_q.append([key[0], key[1]])
            grouped_t.append(attn_type)
            max_q_len = max(max_q_len, key[1] - key[0])
            last_q = key
            last_t = attn_type
            continue
        if key == last_q and attn_type == last_t:
            continue
        offsets.append(idx)
        grouped_q.append([key[0], key[1]])
        grouped_t.append(attn_type)
        max_q_len = max(max_q_len, key[1] - key[0])
        last_q = key
        last_t = attn_type
    offsets.append(int(q_ranges.shape[0]))

    k_cpu = k_ranges.detach().to(device="cpu", dtype=torch.int32).contiguous()
    max_k_len = max((int(end) - int(start) for start, end in k_cpu.tolist()), default=0)

    return (
        torch.tensor(grouped_q, dtype=torch.int32, device=q_ranges.device).contiguous(),
        k_ranges,
        torch.tensor(offsets, dtype=torch.int32, device=q_ranges.device).contiguous(),
        torch.tensor(grouped_t, dtype=torch.int32, device=q_ranges.device).contiguous(),
        int(max_q_len),
        int(max_k_len),
    )


def _flash_lse_to_tq_h(lse, total_q, q_lengths=None):
    if lse is None:
        return None
    if lse.dim() != 2:
        if lse.dim() == 3 and q_lengths is not None and lse.shape[0] == len(q_lengths):
            chunks = []
            for idx, q_len in enumerate(q_lengths):
                q_len = int(q_len)
                if lse.shape[1] == 0 or q_len > lse.shape[2]:
                    return None
                chunks.append(lse[idx, :, :q_len].transpose(0, 1).contiguous())
            merged = torch.cat(chunks, dim=0).float()
            return merged if merged.shape[0] == total_q else None
        return None
    if lse.shape[0] == total_q:
        return lse.float()
    if lse.shape[1] == total_q:
        return lse.transpose(0, 1).contiguous().float()
    return None


def _make_cu_seqlens(lengths, device):
    return torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device=device, dtype=torch.int32)


def _try_flash_segment_merge(
    q,
    k,
    v,
    k_ranges,
    segment_offsets,
    group_q_ranges,
    group_attn_type_map,
    softmax_scale,
):
    if not _flash_segment_fastpath_enabled():
        return None
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        return None
    if group_q_ranges is None or segment_offsets is None or group_attn_type_map is None:
        return None

    flash_attn_varlen = _load_flash_attn_varlen()
    gq_cpu = group_q_ranges.detach().to(device="cpu", dtype=torch.int32).contiguous()
    kr_cpu = k_ranges.detach().to(device="cpu", dtype=torch.int32).contiguous()
    seg_cpu = segment_offsets.detach().to(device="cpu", dtype=torch.int32).contiguous()
    type_cpu = group_attn_type_map.detach().to(device="cpu", dtype=torch.int32).contiguous()

    groups = []
    max_segments = 0
    for group_idx, (q_start, q_end) in enumerate(gq_cpu.tolist()):
        attn_type = int(type_cpu[group_idx].item())
        if attn_type not in (0, 1):
            return None
        seg_start = int(seg_cpu[group_idx].item())
        seg_end = int(seg_cpu[group_idx + 1].item())
        if seg_end <= seg_start or q_end <= q_start:
            return None
        segments = kr_cpu[seg_start:seg_end].tolist()
        max_segments = max(max_segments, len(segments))
        groups.append((int(q_start), int(q_end), attn_type, [(int(a), int(b)) for a, b in segments]))

    if not groups or max_segments == 0:
        return None

    can_pack_full_groups = all(attn_type == 0 or len(segments) == 1 for _, _, attn_type, segments in groups)
    if can_pack_full_groups:
        merged = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)
        covered = torch.zeros((q.shape[0],), device=q.device, dtype=torch.bool)
        for attn_type in (0, 1):
            q_slices = []
            k_slices = []
            v_slices = []
            q_lengths = []
            k_lengths = []
            targets = []
            for q_start, q_end, group_type, segments in groups:
                if group_type != attn_type:
                    continue
                q_slices.append(q[q_start:q_end])
                if attn_type == 0 and len(segments) > 1:
                    k_slices.append(torch.cat([k[start:end] for start, end in segments], dim=0))
                    v_slices.append(torch.cat([v[start:end] for start, end in segments], dim=0))
                    k_lengths.append(sum(end - start for start, end in segments))
                else:
                    k_start, k_end = segments[0]
                    k_slices.append(k[k_start:k_end])
                    v_slices.append(v[k_start:k_end])
                    k_lengths.append(k_end - k_start)
                q_lengths.append(q_end - q_start)
                targets.append((q_start, q_end))
            if not q_slices:
                continue

            out_pass = flash_attn_varlen(
                torch.cat(q_slices, dim=0).contiguous(),
                torch.cat(k_slices, dim=0).contiguous(),
                torch.cat(v_slices, dim=0).contiguous(),
                _make_cu_seqlens(q_lengths, q.device),
                _make_cu_seqlens(k_lengths, q.device),
                int(max(q_lengths)),
                int(max(k_lengths)),
                dropout_p=0.0,
                softmax_scale=float(softmax_scale),
                causal=bool(attn_type == 1),
            )
            if isinstance(out_pass, tuple):
                out_pass = out_pass[0]

            cursor = 0
            for q_start, q_end in targets:
                q_len = q_end - q_start
                merged[q_start:q_end] = out_pass[cursor:cursor + q_len]
                covered[q_start:q_end] = True
                cursor += q_len

        if bool(covered.all().item()):
            return merged

    merged = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    merged_lse = torch.full((q.shape[0], q.shape[1]), -float("inf"), device=q.device, dtype=torch.float32)
    covered = torch.zeros((q.shape[0],), device=q.device, dtype=torch.bool)

    for segment_idx in range(max_segments):
        for attn_type in (0, 1):
            q_slices = []
            k_slices = []
            v_slices = []
            q_lengths = []
            k_lengths = []
            targets = []
            for q_start, q_end, group_type, segments in groups:
                if group_type != attn_type or segment_idx >= len(segments):
                    continue
                k_start, k_end = segments[segment_idx]
                if k_end <= k_start:
                    continue
                q_slices.append(q[q_start:q_end])
                k_slices.append(k[k_start:k_end])
                v_slices.append(v[k_start:k_end])
                q_lengths.append(q_end - q_start)
                k_lengths.append(k_end - k_start)
                targets.append((q_start, q_end))
            if not q_slices:
                continue

            result = flash_attn_varlen(
                torch.cat(q_slices, dim=0).contiguous(),
                torch.cat(k_slices, dim=0).contiguous(),
                torch.cat(v_slices, dim=0).contiguous(),
                _make_cu_seqlens(q_lengths, q.device),
                _make_cu_seqlens(k_lengths, q.device),
                int(max(q_lengths)),
                int(max(k_lengths)),
                dropout_p=0.0,
                softmax_scale=float(softmax_scale),
                causal=bool(attn_type == 1),
                return_attn_probs=True,
            )
            if not isinstance(result, tuple) or len(result) < 2:
                return None
            out_pass = result[0]
            lse_pass = _flash_lse_to_tq_h(result[1], out_pass.shape[0], q_lengths)
            if lse_pass is None:
                return None

            cursor = 0
            for q_start, q_end in targets:
                q_len = q_end - q_start
                out_seg = out_pass[cursor:cursor + q_len].float()
                lse_seg = lse_pass[cursor:cursor + q_len]
                old_lse = merged_lse[q_start:q_end]
                new_lse = torch.maximum(old_lse, lse_seg)
                old_w = torch.exp(old_lse - new_lse)
                seg_w = torch.exp(lse_seg - new_lse)
                denom = (old_w + seg_w).clamp_min(1e-20)
                merged[q_start:q_end] = (
                    merged[q_start:q_end] * old_w.unsqueeze(-1)
                    + out_seg * seg_w.unsqueeze(-1)
                ) / denom.unsqueeze(-1)
                merged_lse[q_start:q_end] = new_lse + torch.log(denom)
                covered[q_start:q_end] = True
                cursor += q_len

    if not bool(covered.all().item()):
        return None
    return merged.to(dtype=q.dtype)


def range_attention(
    q,
    k,
    v,
    q_ranges,
    k_ranges,
    attn_type_map,
    softmax_scale: float,
    *,
    segment_offsets=None,
    group_q_ranges=None,
    group_attn_type_map=None,
    max_q_len=None,
    max_k_len=None,
    flash_cu_seqlens_q=None,
    flash_cu_seqlens_k=None,
    flash_causal=None,
    disjoint_q_ranges=None,
):
    """Run sparse range attention through FlashAttention varlen."""
    del disjoint_q_ranges
    if not q.is_cuda:
        raise RuntimeError("LA Flash range_attention requires CUDA tensors")
    if segment_offsets is None or group_q_ranges is None or group_attn_type_map is None:
        (
            group_q_ranges,
            k_ranges,
            segment_offsets,
            group_attn_type_map,
            computed_max_q_len,
            computed_max_k_len,
        ) = _coalesce_query_groups(q_ranges, k_ranges, attn_type_map)
        if max_q_len is None:
            max_q_len = computed_max_q_len
        if max_k_len is None:
            max_k_len = computed_max_k_len
    elif max_q_len is None:
        lengths = (group_q_ranges[:, 1] - group_q_ranges[:, 0]).detach().to(device="cpu")
        max_q_len = int(lengths.max().item()) if lengths.numel() else 0
    if max_k_len is None:
        k_lengths = (k_ranges[:, 1] - k_ranges[:, 0]).detach().to(device="cpu")
        max_k_len = int(k_lengths.max().item()) if k_lengths.numel() else 0

    if (
        flash_cu_seqlens_q is not None
        and flash_cu_seqlens_k is not None
        and flash_causal is not None
        and _flash_fastpath_enabled()
        and q.dtype in (torch.float16, torch.bfloat16)
        and k.dtype == q.dtype
        and v.dtype == q.dtype
    ):
        flash_attn_varlen = _load_flash_attn_varlen()
        return flash_attn_varlen(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            flash_cu_seqlens_q.contiguous().to(device=q.device, dtype=torch.int32),
            flash_cu_seqlens_k.contiguous().to(device=q.device, dtype=torch.int32),
            int(max_q_len),
            int(max_k_len),
            dropout_p=0.0,
            softmax_scale=float(softmax_scale),
            causal=bool(flash_causal),
        )

    segment_out = _try_flash_segment_merge(
        q,
        k,
        v,
        k_ranges,
        segment_offsets,
        group_q_ranges,
        group_attn_type_map,
        softmax_scale,
    )
    if segment_out is not None:
        return segment_out

    raise RuntimeError(
        "LA Flash could not express this range plan with FlashAttention varlen. "
        "Only attn_type 0/1 range plans are supported in the release path."
    )
