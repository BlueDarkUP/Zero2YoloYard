"""Batched hybrid-mode generation for LocateAnything-3B.

This module keeps the stock hybrid state machine:

  MTP -> error_box -> AR
  AR  -> box_end_ar -> MTP

Rows in a batch may be in different modes. The decode loop therefore stores
per-row KV caches, packs rows with the same mode for one forward call, then
unpacks the clean KV back per row.
"""
import copy
import importlib
import os

import torch


from .hybrid_runtime import (
    ATTN_MODE,
    AR_BATCH_SAN,
    BATCH_SAN,
    DEV,
    N_FUTURE,
    _encode_images,
    _helpers,
    _pad_generated,
    _set_llm_mode,
    _tokenize,
    _tokenize_cached_image,
    build_magi_scheduler_ranges,
    language_model_forward,
    load,
    sample_next_tokens_batched,
    sample_tokens_batched,
)


README_MAX_NEW_TOKENS = 2048
README_TEMPERATURE = 0.7
README_TOP_P = 0.9
README_REPETITION_PENALTY = 1.1

_LAST_HYBRID_STATS = None


def _row_len(kv):
    return kv[0][0].shape[2]


def _pack_stock_kv_rows(kv_rows, rows, dev):
    """Left-pad per-row real-token KV caches for stock-style decoding."""
    lengths = [0 if kv_rows[r] is None else _row_len(kv_rows[r]) for r in rows]
    kmax = max(lengths) if lengths else 0
    if kmax == 0:
        return None, torch.zeros((len(rows), 0), dtype=torch.long, device=dev), lengths, 0

    ref = next(kv_rows[r] for r in rows if kv_rows[r] is not None)
    packed = []
    for layer in range(len(ref)):
        ref_k, ref_v = ref[layer]
        ks, vs = [], []
        for r, length in zip(rows, lengths):
            if length == 0:
                k = ref_k.new_zeros((1, ref_k.shape[1], kmax, ref_k.shape[3]))
                v = ref_v.new_zeros((1, ref_v.shape[1], kmax, ref_v.shape[3]))
            else:
                k, v = kv_rows[r][layer]
                if length < kmax:
                    pad_shape = (1, k.shape[1], kmax - length, k.shape[3])
                    k = torch.cat([k.new_zeros(pad_shape), k], dim=2)
                    v = torch.cat([v.new_zeros(pad_shape), v], dim=2)
            ks.append(k)
            vs.append(v)
        packed.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

    kvalid = torch.zeros((len(rows), kmax), dtype=torch.long, device=dev)
    for i, length in enumerate(lengths):
        if length:
            kvalid[i, kmax - length :] = 1
    return tuple(packed), kvalid, lengths, kmax


def _unpack_stock_after_forward(out_kv, local_row, old_len, uncached_len, kmax, umax):
    """Keep old real KV plus the right-aligned uncached real tokens; drop pads/window."""
    out = []
    u0 = kmax + (umax - uncached_len)
    u1 = kmax + umax
    for k, v in out_kv:
        parts_k, parts_v = [], []
        if old_len:
            parts_k.append(k[local_row : local_row + 1, :, kmax - old_len : kmax, :])
            parts_v.append(v[local_row : local_row + 1, :, kmax - old_len : kmax, :])
        if uncached_len:
            parts_k.append(k[local_row : local_row + 1, :, u0:u1, :])
            parts_v.append(v[local_row : local_row + 1, :, u0:u1, :])
        out.append((torch.cat(parts_k, dim=2).contiguous(),
                    torch.cat(parts_v, dim=2).contiguous()))
    return tuple(out)


def _mk_generate_kwargs(temperature, top_p, top_k, repetition_penalty, row_temp=None):
    t = temperature if row_temp is None else row_temp
    gk = {"repetition_penalty": repetition_penalty, "generation_mode": "hybrid"}
    if t and t > 0:
        gk["temperature"] = t
    if top_p is not None:
        gk["top_p"] = top_p
    if top_k is not None:
        gk["top_k"] = top_k
    return gk


def _classify_ar_token(token_val, tids):
    if token_val == tids["box_end_token_id"]:
        return "box_end_ar"
    if tids["coord_start_token_id"] <= token_val <= tids["coord_end_token_id"]:
        return "coord_ar"
    if token_val == tids["none_token_id"]:
        return "coord_ar"
    return "im_end"


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() not in {"0", "false", "no", "off", ""}


def _env_int(name, default):
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return int(val)


def _kv_pack_token_budget():
    return max(0, _env_int("LA_FLASH_KV_PACK_TOKEN_BUDGET", 0))


def _debug_enabled(debug):
    return _env_flag("LA_FLASH_DEBUG", False) if debug is None else bool(debug)


def _new_hybrid_stats(total_rows, scheduler, group_size, hold_max_steps, adaptive_hold_mtp_max=0):
    return {
        "scheduler": scheduler,
        "requested_group_size": int(group_size or 0),
        "hold_max_steps": int(hold_max_steps),
        "adaptive_hold_mtp_max": int(adaptive_hold_mtp_max),
        "input_batches": 1,
        "input_rows": int(total_rows),
        "groups": 0,
        "group_sizes": [],
        "decode_loops": 0,
        "mixed_mode_cycles": 0,
        "eager_mtp_then_ar_cycles": 0,
        "ar_first_cycles": 0,
        "pipeline_ar_after_mtp_cycles": 0,
        "adaptive_hold_cycles": 0,
        "adaptive_ar_first_cycles": 0,
        "hold_ar_steps": 0,
        "hold_ar_held_mtp_rows": 0,
        "hold_ar_limit_mtp_forwards": 0,
        "mtp_forwards": 0,
        "ar_forwards": 0,
        "mtp_forward_rows": 0,
        "ar_forward_rows": 0,
        "mtp_forward_query_tokens": 0,
        "ar_forward_query_tokens": 0,
        "max_mtp_forward_rows": 0,
        "max_ar_forward_rows": 0,
        "mtp_max_uncached_len": 0,
        "ar_max_uncached_len": 0,
        "mtp_forward_row_hist": {},
        "ar_forward_row_hist": {},
        "prompt_prefill_mode": _hybrid_prefill_mode(),
        "prompt_prefill_forwards": 0,
        "prompt_prefill_forward_rows": 0,
        "prompt_prefill_forward_query_tokens": 0,
        "prompt_prefill_real_tokens": 0,
        "prompt_prefill_shared_groups": 0,
        "prompt_prefill_shared_rows": 0,
        "prompt_prefill_shared_saved_tokens": 0,
        "kv_bucket_splits": 0,
        "kv_bucket_groups": 0,
        "kv_bucket_max_packed_tokens": 0,
    }


def _set_last_hybrid_stats(stats):
    global _LAST_HYBRID_STATS
    _LAST_HYBRID_STATS = copy.deepcopy(stats) if stats is not None else None


def get_last_hybrid_stats():
    """Return scheduler/forward statistics from the most recent hybrid batch."""
    return copy.deepcopy(_LAST_HYBRID_STATS)


def _record_group_stats(stats, bsz):
    if stats is None:
        return
    stats["groups"] += 1
    stats["group_sizes"].append(int(bsz))


def _bump_hist(hist, val):
    key = str(int(val))
    hist[key] = int(hist.get(key, 0)) + 1


def _record_forward_stats(stats, kind, rows, q_len, uncached_lens):
    if stats is None:
        return
    prefix = "mtp" if kind == "mtp" else "ar"
    nrows = int(len(rows))
    q_len = int(q_len)
    stats[f"{prefix}_forwards"] += 1
    stats[f"{prefix}_forward_rows"] += nrows
    stats[f"{prefix}_forward_query_tokens"] += nrows * q_len
    stats[f"max_{prefix}_forward_rows"] = max(stats[f"max_{prefix}_forward_rows"], nrows)
    stats[f"{prefix}_max_uncached_len"] = max(
        stats[f"{prefix}_max_uncached_len"],
        max((int(x) for x in uncached_lens), default=0),
    )
    _bump_hist(stats[f"{prefix}_forward_row_hist"], nrows)


def _record_prefill_stats(stats, rows, q_len, real_tokens, shared_groups=0, shared_rows=0, saved_tokens=0):
    if stats is None:
        return
    nrows = int(rows)
    stats["prompt_prefill_forwards"] += 1
    stats["prompt_prefill_forward_rows"] += nrows
    stats["prompt_prefill_forward_query_tokens"] += nrows * int(q_len)
    stats["prompt_prefill_real_tokens"] += int(real_tokens)
    stats["prompt_prefill_shared_groups"] += int(shared_groups)
    stats["prompt_prefill_shared_rows"] += int(shared_rows)
    stats["prompt_prefill_shared_saved_tokens"] += int(saved_tokens)


def _split_rows_by_kv_budget(rows, kv_rows):
    """Keep dense left-padded KV packs bounded when a few rows become long tails."""
    budget = _kv_pack_token_budget()
    if budget <= 0 or len(rows) <= 1:
        return [rows]
    lengths = [0 if kv_rows[r] is None else _row_len(kv_rows[r]) for r in rows]
    if not lengths or max(lengths) * len(rows) <= budget:
        return [rows]

    groups = []
    current = []
    current_max = 0
    for row, length in sorted(zip(rows, lengths), key=lambda item: item[1]):
        next_max = max(current_max, int(length))
        if current and next_max * (len(current) + 1) > budget:
            groups.append(current)
            current = [row]
            current_max = int(length)
        else:
            current.append(row)
            current_max = next_max
    if current:
        groups.append(current)
    return groups or [rows]


def _record_kv_bucket_stats(stats, groups, kv_rows):
    if stats is None:
        return
    max_packed = 0
    for group in groups:
        if not group:
            continue
        kmax = max((0 if kv_rows[r] is None else _row_len(kv_rows[r])) for r in group)
        max_packed = max(max_packed, int(kmax) * len(group))
    stats["kv_bucket_max_packed_tokens"] = max(stats["kv_bucket_max_packed_tokens"], max_packed)
    if len(groups) > 1:
        stats["kv_bucket_splits"] += 1
        stats["kv_bucket_groups"] += len(groups)


def _hybrid_scheduler(scheduler):
    val = os.environ.get("LA_FLASH_HYBRID_SCHEDULER", "eager") if scheduler is None else scheduler
    val = str(val).strip().lower()
    aliases = {
        "": "eager",
        "default": "eager",
        "normal": "eager",
        "hold": "hold_ar",
        "hold-ar": "hold_ar",
        "hold_mtp": "hold_ar",
        "hold-mtp": "hold_ar",
        "repair_first": "ar_first",
        "repair-first": "ar_first",
        "ar-first": "ar_first",
    }
    val = aliases.get(val, val)
    if val not in {"eager", "hold_ar", "ar_first", "pipeline", "adaptive"}:
        raise ValueError("scheduler must be one of: eager, hold_ar, ar_first, pipeline, adaptive")
    return val


def _hybrid_group_size(group_size):
    if group_size is None:
        return max(0, _env_int("LA_FLASH_HYBRID_GROUP_SIZE", 0))
    return max(0, int(group_size))


def _hybrid_prefill_mode():
    val = os.environ.get("LA_FLASH_HYBRID_PREFILL", "shared").strip().lower()
    aliases = {
        "0": "none",
        "false": "none",
        "off": "none",
        "legacy": "none",
        "1": "per_row",
        "true": "per_row",
        "on": "per_row",
        "single": "per_row",
        "row": "per_row",
        "rows": "per_row",
        "batched": "batch",
        "prefix": "shared",
        "shared_prefix": "shared",
        "shared-image": "shared",
        "shared_image": "shared",
        "vision": "shared",
    }
    val = aliases.get(val, val)
    if val not in {"none", "per_row", "batch", "shared"}:
        raise ValueError("LA_FLASH_HYBRID_PREFILL must be one of none, per_row, batch, shared")
    return val


def _tolist(t):
    return t.detach().cpu().tolist()


def _safe_decode_rows(tok, input_ids):
    rows = []
    for row in _tolist(input_ids):
        try:
            rows.append(tok.decode(torch.tensor(row), skip_special_tokens=False))
        except Exception:
            rows.append("<decode failed>")
    return rows


def _safe_decode_row(tok, row):
    try:
        return tok.decode(torch.tensor(row), skip_special_tokens=False)
    except Exception:
        return "<decode failed>"


def _effective_allowed_mask(mask2d, q_len, past_len, mtp_window=False):
    """Readable 1/0 q-by-k mask derived from the 2D key-valid mask.

    This mirrors the model path at a high level:
    causal + padding columns, then the MTP window update
    attn[-block:, -block:] = visible and attn[-block:, -block-1] = masked.
    """
    rows = []
    key_valid = mask2d.detach().cpu().bool()
    total_len = int(key_valid.numel())
    for qi in range(q_len):
        q_abs = past_len + qi
        row = []
        for ki in range(total_len):
            row.append(1 if bool(key_valid[ki]) and ki <= q_abs else 0)
        rows.append(row)

    if mtp_window and q_len >= N_FUTURE and total_len >= N_FUTURE:
        q0 = q_len - N_FUTURE
        k0 = total_len - N_FUTURE
        for qi in range(q0, q_len):
            for ki in range(k0, total_len):
                rows[qi][ki] = 1
            if k0 - 1 >= 0:
                rows[qi][k0 - 1] = 0
    return rows


def _tail_matrix(mat, rows=None, cols=None):
    if rows is not None:
        mat = mat[-rows:]
    if cols is not None:
        mat = [row[-cols:] for row in mat]
    return mat


def _format_01_matrix(mat):
    return "\n".join("  " + " ".join(str(int(v)) for v in row) for row in mat)


def _safe_sdpa_mask_enabled():
    return _env_flag("LA_FLASH_SDPA_SAFE_4D_MASK", True)


def _build_safe_sdpa_visible_mask(attention_mask_2d, input_ids, past_len, mtp_window=False):
    """Build a 4D 1/0 visible mask, with harmless visibility for all-masked pad queries.

    The remote Qwen2 SDPA path uses a 2D key-valid mask and can create fully
    masked query rows for left-padded, no-cache prefill. Those rows can produce
    NaNs inside SDPA and later contaminate real tokens through masked K columns.
    This 4D mask keeps real-token visibility identical, and only gives otherwise
    all-masked query rows one valid fallback key so their activations stay finite.
    """
    bsz, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    key_len = int(attention_mask_2d.shape[1])
    dev = input_ids.device
    key_valid = attention_mask_2d.to(dtype=torch.bool, device=dev)
    key_idx = torch.arange(key_len, device=dev).view(1, 1, key_len)
    q_abs = (past_len + torch.arange(q_len, device=dev)).view(1, q_len, 1)
    visible = key_valid[:, None, :] & (key_idx <= q_abs)

    if mtp_window and q_len >= N_FUTURE and key_len >= N_FUTURE:
        k0 = key_len - N_FUTURE
        visible[:, -N_FUTURE:, k0:key_len] = key_valid[:, None, k0:key_len]
        blocked_k = k0 - 1
        if blocked_k >= 0:
            visible[:, -N_FUTURE:, blocked_k] = False

    row_has_key = visible.any(dim=-1)
    fallback_rows = int((~row_has_key).sum().item())
    if fallback_rows:
        for b in range(bsz):
            valid = torch.nonzero(key_valid[b], as_tuple=False).flatten()
            fallback = int(valid[0].item()) if valid.numel() else 0
            missing = torch.nonzero(~row_has_key[b], as_tuple=False).flatten()
            if missing.numel():
                visible[b, missing, fallback] = True

    mask = visible[:, None, :, :].to(dtype=torch.bfloat16)
    try:
        mask._la_flash_visible_mask = True
    except Exception:
        pass
    return mask, fallback_rows


def _mask_desc(mask):
    if mask is None:
        return "none"
    if isinstance(mask, dict):
        return "magi_ranges"
    if hasattr(mask, "dim"):
        return "4d_safe_sdpa" if mask.dim() == 4 else "2d_key_valid"
    return type(mask).__name__


def _forward_attention_mask(model, input_ids, attention_mask_2d, past_len, mtp_window=False, range_plan=False):
    llm = model.language_model.model
    if getattr(model, "_la_flash_requested_attn", ATTN_MODE) in {"magi", "la_flash"}:
        range_plan = build_magi_scheduler_ranges(
            model, attention_mask_2d, input_ids, past_len, mtp_window=mtp_window)
        if range_plan is not None:
            return range_plan, 0
    needs_safe_pad = (
        past_len == 0
        and attention_mask_2d is not None
        and attention_mask_2d.dim() == 2
        and input_ids.shape[0] > 1
    )
    if (
        getattr(llm, "_attn_implementation", None) == "sdpa"
        and _safe_sdpa_mask_enabled()
        and needs_safe_pad
        and attention_mask_2d is not None
        and attention_mask_2d.dim() == 2
    ):
        return _build_safe_sdpa_visible_mask(attention_mask_2d, input_ids, past_len, mtp_window)
    return attention_mask_2d, 0


def _actual_sdpa_allowed_masks(model, input_ids, attention_mask, past_len):
    """Recreate the remote Qwen2 SDPA 4D additive mask and return a 0/1 view."""
    llm = model.language_model.model
    mod = importlib.import_module(type(llm).__module__)
    bsz, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    dummy = torch.empty(
        (bsz, q_len, 1),
        dtype=torch.bfloat16,
        device=input_ids.device,
    )
    mask4 = mod._prepare_4d_causal_attention_mask(
        attention_mask,
        (bsz, q_len),
        dummy,
        past_len,
        sliding_window=getattr(llm.config, "sliding_window", None),
    )
    remote_ar_decode = q_len == 1 or (
        input_ids is not None and int(input_ids[0, -1].item()) != int(llm.text_mask_token_id)
    )
    if not remote_ar_decode and mask4 is not None and mask4.dim() == 4:
        rows = []
        for b in range(bsz):
            rows.append(
                mod.update_causal_mask_for_one_gen_window_2d(
                    input_ids[b],
                    mask4[b][0].clone(),
                    block_size=int(llm.block_size),
                    use_cache=True,
                    causal_attn=bool(getattr(llm, "causal_attn", False)),
                ).unsqueeze(0)
            )
        mask4 = torch.stack(rows, dim=0)
    allowed = (mask4[:, 0] >= 0).to(torch.int8).detach().cpu().tolist()
    return allowed, tuple(mask4.shape), remote_ar_decode


def _debug_magi_ranges(q_len, past_len, mtp_window=False):
    kv_len = past_len + q_len
    ar_decode = not mtp_window
    if ar_decode:
        return {
            "q_ranges": [[0, q_len]],
            "k_ranges": [[0, kv_len]],
            "attn_type_map": ["CAUSAL"],
        }

    block = N_FUTURE
    if not (0 < block <= q_len <= kv_len):
        return {"error": f"invalid magi MTP shape: block={block}, q_len={q_len}, kv_len={kv_len}"}

    prefix_len = kv_len - block
    blocked_k = prefix_len - 1
    q_ranges, k_ranges, attn_types = [], [], []
    if q_len == kv_len:
        if prefix_len > 0:
            q_ranges.append([0, prefix_len])
            k_ranges.append([0, prefix_len])
            attn_types.append("CAUSAL")
        if prefix_len > 0 and blocked_k > 0:
            q_ranges.append([prefix_len, kv_len])
            k_ranges.append([0, blocked_k])
            attn_types.append("FULL")
        q_ranges.append([prefix_len, kv_len])
        k_ranges.append([prefix_len, kv_len])
        attn_types.append("FULL")
    else:
        recompute = q_len - block
        q_global_start = kv_len - q_len
        for i in range(recompute):
            g = q_global_start + i
            q_ranges.append([i, i + 1])
            k_ranges.append([0, g + 1])
            attn_types.append("FULL")
        q_win = [recompute, q_len]
        if blocked_k > 0:
            q_ranges.append(q_win)
            k_ranges.append([0, blocked_k])
            attn_types.append("FULL")
        q_ranges.append(q_win)
        k_ranges.append([prefix_len, kv_len])
        attn_types.append("FULL")

    return {"q_ranges": q_ranges, "k_ranges": k_ranges, "attn_type_map": attn_types}


def _print_debug_forward(label, model, tok, input_ids, attention_mask, position_ids,
                         past_len, mtp_window=False, extra=None, attention_impl="sdpa"):
    print(f"\n========== LA Flash DEBUG {label} ==========", flush=True)
    if extra:
        for k, v in extra.items():
            print(f"{k}: {v}", flush=True)
    tail = int(os.environ.get("LA_FLASH_DEBUG_TAIL", "15"))
    bsz, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    key_len = int(attention_mask.shape[1])
    q_tail, k_tail = min(tail, q_len), min(tail, key_len)
    print(
        "shapes: "
        f"input_ids={tuple(input_ids.shape)} "
        f"position_ids={tuple(position_ids.shape)} "
        f"attention_mask_key_valid={tuple(attention_mask.shape)} "
        f"mask_2d_q_by_k=({bsz}, {q_len}, {key_len}) "
        f"mask_2d_tail=({bsz}, {q_tail}, {k_tail}) "
        f"past_len={past_len} q_len={q_len} "
        f"mtp_window={mtp_window} ar_decode={not mtp_window}",
        flush=True,
    )
    print(f"dtypes/devices: input_ids={input_ids.dtype}@{input_ids.device} position_ids={position_ids.dtype}@{position_ids.device} attention_mask={attention_mask.dtype}@{attention_mask.device}", flush=True)
    print(f"attention_impl={attention_impl}", flush=True)
    input_rows = _tolist(input_ids)
    pos_rows = _tolist(position_ids)
    print(f"tail_window_last={tail}", flush=True)
    print(f"input_ids_tail.shape=({bsz}, {q_tail})", flush=True)
    print(f"position_ids_tail.shape=({bsz}, {q_tail})", flush=True)
    actual_sdpa = None
    if attention_impl in {"sdpa", "eager", "la_flash"}:
        try:
            actual_sdpa = _actual_sdpa_allowed_masks(model, input_ids, attention_mask, past_len)
            print(
                f"actual_sdpa_4d_mask_shape={actual_sdpa[1]} "
                f"remote_ar_decode={actual_sdpa[2]}",
                flush=True,
            )
        except Exception as e:
            print(f"actual_sdpa_4d_mask_debug_failed={type(e).__name__}: {e}", flush=True)

    for b in range(input_ids.shape[0]):
        ids_tail = input_rows[b][-tail:]
        pos_tail = pos_rows[b][-tail:]
        allowed = _effective_allowed_mask(attention_mask[b], input_ids.shape[1], past_len, mtp_window)
        q_tail = min(tail, len(allowed))
        k_tail = min(tail, len(allowed[0]) if allowed else 0)
        allowed_tail = _tail_matrix(allowed, rows=q_tail, cols=k_tail)
        print(f"batch_row={b} ar_decode={not mtp_window}", flush=True)
        print(f"input_ids_tail[-{tail}:]: {ids_tail}", flush=True)
        print(f"decoded_tail[-{tail}:]: {_safe_decode_row(tok, ids_tail)}", flush=True)
        print(f"position_ids_tail[-{tail}:]: {pos_tail}", flush=True)
        print(f"expected_mask_2d_tail[-{q_tail}:,-{k_tail}:].shape=({q_tail}, {k_tail})", flush=True)
        print(_format_01_matrix(allowed_tail), flush=True)
        if actual_sdpa is not None:
            actual = actual_sdpa[0][b]
            actual_tail = _tail_matrix(actual, rows=q_tail, cols=k_tail)
            mismatch = sum(
                int(allowed[qi][ki] != actual[qi][ki])
                for qi in range(len(allowed))
                for ki in range(len(allowed[qi]))
            )
            print(
                f"actual_sdpa_mask_2d_tail[-{q_tail}:,-{k_tail}:].shape=({q_tail}, {k_tail})",
                flush=True,
            )
            print(_format_01_matrix(actual_tail), flush=True)
            print(f"expected_vs_actual_sdpa_mismatch_count={mismatch}", flush=True)

    if _env_flag("LA_FLASH_DEBUG_FULL_MASK", False):
        masks = [
            _effective_allowed_mask(attention_mask[b], input_ids.shape[1], past_len, mtp_window)
            for b in range(input_ids.shape[0])
        ]
        print("effective_allowed_mask_q_by_k_FULL:", masks, flush=True)
    if attention_impl == "magi":
        if bsz == 1:
            print(
                "magi_ranges:",
                _debug_magi_ranges(input_ids.shape[1], past_len, mtp_window),
                flush=True,
            )
        else:
            print(
                "magi_ranges: built once per forward from the batched scheduler mask",
                flush=True,
            )
            print(
                "magi_ranges_single_row_template:",
                _debug_magi_ranges(input_ids.shape[1], past_len, mtp_window),
                flush=True,
            )


def _common_prefix_len(prompt_ids, rows):
    if not rows:
        return 0
    first = prompt_ids[rows[0]]
    max_len = min(int(prompt_ids[r].numel()) for r in rows)
    prefix_len = 0
    for idx in range(max_len):
        val = int(first[idx].item())
        if all(int(prompt_ids[r][idx].item()) == val for r in rows[1:]):
            prefix_len += 1
        else:
            break
    return prefix_len


def _prefill_shared_prefix_kv_rows(model, prompt_ids, vit_list, img_tok, pad, dev, stats=None, debug=False):
    """Cache one common prompt prefix per image-feature group.

    Multi-category split repeats the same image feature tensor for each
    category prompt.  Token ids are identical through the image tokens and the
    fixed prompt prefix, so we prefill that shared prefix once and let each
    category row forward only its text suffix.
    """
    bsz = len(prompt_ids)
    kv_rows = [None] * bsz
    cached_lens = [0] * bsz
    groups = {}
    for row, vit in enumerate(vit_list):
        groups.setdefault(id(vit), []).append(row)

    items = []
    min_prefix_len = max(1, _env_int("LA_FLASH_SHARED_PREFILL_MIN_PREFIX", 64))
    for rows in groups.values():
        if len(rows) < 2:
            continue
        prefix_len = _common_prefix_len(prompt_ids, rows)
        if prefix_len < min_prefix_len:
            continue
        prefix_ids = prompt_ids[rows[0]][:prefix_len]
        image_token_count = int((prefix_ids == img_tok).sum().item())
        if image_token_count != int(vit_list[rows[0]].shape[0]):
            if debug:
                print(
                    "LA Flash shared prefill skip group: "
                    f"rows={rows} prefix_len={prefix_len} "
                    f"image_tokens={image_token_count} visual_rows={int(vit_list[rows[0]].shape[0])}",
                    flush=True,
                )
            continue
        items.append((rows, prefix_ids, vit_list[rows[0]]))

    if not items:
        return kv_rows, cached_lens

    lengths = [int(ids.numel()) for _rows, ids, _vit in items]
    pmax = max(lengths)
    input_ids = torch.full((len(items), pmax), pad, dtype=torch.long, device=dev)
    amask = torch.zeros((len(items), pmax), dtype=torch.long, device=dev)
    pos = torch.ones((len(items), pmax), dtype=torch.long, device=dev)
    for item_idx, (_rows, ids, _vit) in enumerate(items):
        length = lengths[item_idx]
        left = pmax - length
        input_ids[item_idx, left:] = ids.to(dev)
        amask[item_idx, left:] = 1
        pos[item_idx, left:] = torch.arange(length, dtype=torch.long, device=dev)

    visual_features = torch.cat([vit for _rows, _ids, vit in items], dim=0)
    assert int((input_ids == img_tok).sum().item()) == visual_features.shape[0], \
        "shared-prefix image-token count != supplied visual_features rows"

    if debug:
        group_sizes = [len(rows) for rows, _ids, _vit in items]
        print(
            "LA Flash hybrid shared prompt prefill "
            f"groups={len(items)} group_sizes={group_sizes} prefix_lens={lengths}",
            flush=True,
        )

    forward_mask, fallback_rows = _forward_attention_mask(
        model, input_ids, amask, 0, mtp_window=False)
    if debug and fallback_rows:
        print(
            "LA Flash hybrid shared prefill safe SDPA fallback "
            f"query_rows={fallback_rows}",
            flush=True,
        )
    forward_kwargs = dict(
        input_ids=input_ids,
        visual_features=visual_features,
        image_token_index=img_tok,
        attention_mask=forward_mask,
        position_ids=pos,
        past_key_values=None,
        use_cache=True,
    )
    if isinstance(forward_mask, dict):
        out = language_model_forward(model, **forward_kwargs, return_logits=False)
    else:
        out = model.language_model.model(**forward_kwargs)

    real_tokens = sum(lengths)
    shared_rows = sum(len(rows) for rows, _ids, _vit in items)
    saved_tokens = sum((len(rows) - 1) * length for (rows, _ids, _vit), length in zip(items, lengths))
    _record_prefill_stats(
        stats,
        rows=len(items),
        q_len=pmax,
        real_tokens=real_tokens,
        shared_groups=len(items),
        shared_rows=shared_rows,
        saved_tokens=saved_tokens,
    )

    for item_idx, (rows, _ids, _vit) in enumerate(items):
        prefix_len = lengths[item_idx]
        prefix_kv = _unpack_stock_after_forward(out.past_key_values, item_idx, 0, prefix_len, 0, pmax)
        for row in rows:
            kv_rows[row] = prefix_kv
            cached_lens[row] = prefix_len

    return kv_rows, cached_lens


@torch.no_grad()
def _prefill_prompt_kv_rows(model, prompt_ids, vit_list, img_tok, pad, dev, mode, debug=False, stats=None):
    """Return per-row prompt KV caches and cached lengths.

    ``mode='none'`` preserves the legacy stock-like first MTP forward where the
    whole prompt and the 6-token MTP window are forwarded together.  The split
    prefill modes keep prompt KV clean before the scheduler batches only short
    suffix/window forwards, which avoids ragged prompt+window masking in the
    first decode step.
    """
    bsz = len(prompt_ids)
    lengths = [int(p.numel()) for p in prompt_ids]
    if mode == "none":
        return [None] * bsz, [0] * bsz

    base = model.language_model.model
    if debug:
        print(f"LA Flash hybrid prompt prefill mode={mode} rows={bsz} lengths={lengths}", flush=True)

    if mode == "shared":
        return _prefill_shared_prefix_kv_rows(
            model, prompt_ids, vit_list, img_tok, pad, dev, stats=stats, debug=debug)

    if mode == "per_row":
        kv_rows = []
        for b, ids in enumerate(prompt_ids):
            ids = ids.to(dev).unsqueeze(0)
            pos = torch.arange(ids.shape[1], dtype=torch.long, device=dev).unsqueeze(0)
            out = base(
                input_ids=ids,
                visual_features=vit_list[b],
                image_token_index=img_tok,
                attention_mask=None,
                position_ids=pos,
                past_key_values=None,
                use_cache=True,
            )
            kv_rows.append(out.past_key_values)
            _record_prefill_stats(stats, rows=1, q_len=ids.shape[1], real_tokens=ids.shape[1])
        return kv_rows, lengths

    pmax = max(lengths)
    input_ids = torch.full((bsz, pmax), pad, dtype=torch.long, device=dev)
    amask = torch.zeros((bsz, pmax), dtype=torch.long, device=dev)
    pos = torch.ones((bsz, pmax), dtype=torch.long, device=dev)
    for b, ids in enumerate(prompt_ids):
        left = pmax - lengths[b]
        input_ids[b, left:] = ids.to(dev)
        amask[b, left:] = 1
        pos[b, left:] = torch.arange(lengths[b], dtype=torch.long, device=dev)

    visual_features = torch.cat(vit_list, dim=0)
    assert int((input_ids == img_tok).sum().item()) == visual_features.shape[0], \
        "image-token count != supplied visual_features rows"
    forward_mask, fallback_rows = _forward_attention_mask(
        model, input_ids, amask, 0, mtp_window=False)
    if debug and fallback_rows:
        print(
            "LA Flash hybrid batch prefill safe SDPA fallback "
            f"query_rows={fallback_rows}",
            flush=True,
        )
    forward_kwargs = dict(
        input_ids=input_ids,
        visual_features=visual_features,
        image_token_index=img_tok,
        attention_mask=forward_mask,
        position_ids=pos,
        past_key_values=None,
        use_cache=True,
    )
    if isinstance(forward_mask, dict):
        out = language_model_forward(model, **forward_kwargs, return_logits=False)
    else:
        out = base(**forward_kwargs)
    _record_prefill_stats(stats, rows=bsz, q_len=pmax, real_tokens=sum(lengths))
    kv_rows = [
        _unpack_stock_after_forward(out.past_key_values, b, 0, lengths[b], 0, pmax)
        for b in range(bsz)
    ]
    return kv_rows, lengths


@torch.no_grad()
def generate_batch_hybrid(pairs, temperature=README_TEMPERATURE, top_p=README_TOP_P, top_k=None,
                          repetition_penalty=README_REPETITION_PENALTY,
                          max_new_tokens=README_MAX_NEW_TOKENS, temps=None,
                          debug=None, scheduler=None, group_size=None,
                          vision_features=None, _stats=None):
    """Batched stock-style LocateAnything-3B hybrid generation.

    This mirrors ``model.generate(..., generation_mode='hybrid')``: each row
    owns a full ``generated`` token stream plus a KV cache truncated to real
    generated tokens before sampling.  MTP forwards
    ``generated[cached_len:] + duplicate-last + mask*5``; AR forwards
    ``generated[cached_len:]``.
    """
    tok, _, model = load()
    san, hpat = _helpers()
    tids = model.token_ids
    img_tok = model.config.image_token_index
    mask_tok = tids["default_mask_token_id"]
    im_end = tids["im_end_token_id"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else im_end
    dev = DEV

    if not pairs:
        return []
    if temps is not None and len(temps) != len(pairs):
        raise ValueError("temps must have the same length as pairs")
    if vision_features is not None and len(vision_features) != len(pairs):
        raise ValueError("vision_features must have the same length as pairs")
    debug = _debug_enabled(debug)
    scheduler = _hybrid_scheduler(scheduler)
    group_size = _hybrid_group_size(group_size)
    requested_attn = getattr(model, "_la_flash_requested_attn", ATTN_MODE)
    use_magi = requested_attn == "magi"
    prefill_mode = _hybrid_prefill_mode()
    hold_max_steps = max(0, _env_int("LA_FLASH_HYBRID_HOLD_MAX_STEPS", 5))
    adaptive_hold_mtp_max = max(0, _env_int("LA_FLASH_HYBRID_ADAPTIVE_HOLD_MTP_MAX", 3))
    top_level_stats = _stats is None
    if top_level_stats:
        _stats = _new_hybrid_stats(
            len(pairs), scheduler, group_size, hold_max_steps, adaptive_hold_mtp_max)
        if os.environ.get("LA_FLASH_PLAN_STATS", "0") == "1":
            model._la_flash_sparse_plan_stats = None
    if group_size and len(pairs) > group_size:
        outs = []
        if debug:
            print(
                f"LA Flash hybrid grouped scheduling: total_rows={len(pairs)} "
                f"group_size={group_size} scheduler={scheduler} hold_max_steps={hold_max_steps} "
                f"adaptive_hold_mtp_max={adaptive_hold_mtp_max}",
                flush=True,
            )
        for start in range(0, len(pairs), group_size):
            end = min(start + group_size, len(pairs))
            chunk_temps = temps[start:end] if temps is not None else None
            chunk_vision_features = (
                vision_features[start:end] if vision_features is not None else None
            )
            if debug:
                print(f"LA Flash hybrid group rows=[{start}:{end}]", flush=True)
            outs.extend(generate_batch_hybrid(
                pairs[start:end],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                temps=chunk_temps,
                debug=debug,
                scheduler=scheduler,
                group_size=0,
                vision_features=chunk_vision_features,
                _stats=_stats,
            ))
        if top_level_stats:
            _set_last_hybrid_stats(_stats)
        return outs

    use_cached_tokenize = (
        vision_features is not None
        and os.environ.get("LA_FLASH_CACHE_TOKENIZE", "1") != "0"
    )
    if use_cached_tokenize:
        try:
            prompt_ids = [
                _tokenize_cached_image(q, int(v.shape[0]), im=im)
                for (im, q), v in zip(pairs, vision_features)
            ]
        except Exception as exc:
            if os.environ.get("LA_FLASH_CACHE_TOKENIZE_STRICT", "0") == "1":
                raise
            if debug:
                print(f"LA Flash cached tokenize fallback: {exc}", flush=True)
            prompt_ids = [_tokenize(im, q) for im, q in pairs]
    else:
        prompt_ids = [_tokenize(im, q) for im, q in pairs]
    vit_list = (
        list(vision_features)
        if vision_features is not None
        else _encode_images([im for im, _ in pairs])
    )
    lengths = [int(p.numel()) for p in prompt_ids]
    bsz = len(pairs)
    _record_group_stats(_stats, bsz)

    _set_llm_mode(model, requested_attn)

    modes = ["mtp"] * bsz
    finished = [False] * bsz
    gen_ids = [[] for _ in range(bsz)]
    full_ids = [list(ids.detach().cpu().tolist()) for ids in prompt_ids]
    kv_rows, cached_lens = _prefill_prompt_kv_rows(
        model, prompt_ids, vit_list, img_tok, pad, dev, prefill_mode, debug=debug, stats=_stats)
    total_limits = [lengths[b] + max_new_tokens for b in range(bsz)]

    row_temps = [float(temperature or 0.0)] * bsz if temps is None else [float(t or 0.0) for t in temps]

    def run_ar(ar_rows, step_idx):
        row_groups = _split_rows_by_kv_budget(ar_rows, kv_rows)
        _record_kv_bucket_stats(_stats, row_groups, kv_rows)
        for row_group in row_groups:
            _step_stock_ar_rows(
                model, san, tids, prompt_ids, kv_rows, row_group,
                cached_lens, full_ids, gen_ids, modes, finished, total_limits,
                pad, img_tok, row_temps, temperature, top_p, top_k,
                repetition_penalty, dev, tok, debug, step_idx, use_magi, _stats,
            )

    def run_mtp(mtp_rows, step_idx):
        if any(cached_lens[r] == 0 for r in mtp_rows) and any(cached_lens[r] > 0 for r in mtp_rows):
            first_rows = [r for r in mtp_rows if cached_lens[r] == 0]
            cached_rows = [r for r in mtp_rows if cached_lens[r] > 0]
            if first_rows:
                run_mtp(first_rows, step_idx)
            if cached_rows:
                run_mtp(cached_rows, step_idx)
            return
        row_groups = _split_rows_by_kv_budget(mtp_rows, kv_rows)
        _record_kv_bucket_stats(_stats, row_groups, kv_rows)
        if len(row_groups) > 1:
            for row_group in row_groups:
                run_mtp(row_group, step_idx)
            return
        _step_stock_mtp_rows(
            model, san, hpat, tids, prompt_ids, kv_rows, mtp_rows,
            cached_lens, full_ids, gen_ids, modes, finished, total_limits,
            vit_list, pad, mask_tok, img_tok, row_temps, top_p, top_k,
            repetition_penalty, dev, tok, debug, step_idx, use_magi, _stats,
        )

    def live_rows(mode):
        return [b for b in range(bsz) if not finished[b] and modes[b] == mode]

    step = 0
    hold_steps = 0
    while not all(finished) and step <= max_new_tokens:
        step += 1
        if _stats is not None:
            _stats["decode_loops"] += 1
        if scheduler == "hold_ar" and hold_max_steps > 0:
            ar_rows = live_rows("ar")
            mtp_rows = live_rows("mtp")
            if ar_rows and mtp_rows and _stats is not None:
                _stats["mixed_mode_cycles"] += 1
            if ar_rows and (hold_steps < hold_max_steps or not mtp_rows):
                if mtp_rows and _stats is not None:
                    _stats["hold_ar_steps"] += 1
                    _stats["hold_ar_held_mtp_rows"] += len(mtp_rows)
                run_ar(ar_rows, step)
                hold_steps += 1
                continue
            if mtp_rows:
                if ar_rows and _stats is not None:
                    _stats["hold_ar_limit_mtp_forwards"] += 1
                run_mtp(mtp_rows, step)
                hold_steps = 0
            continue

        if scheduler in {"ar_first", "pipeline", "adaptive"}:
            ar_rows_at_loop_start = live_rows("ar")
            mtp_rows_at_loop_start = live_rows("mtp")
            mixed = bool(ar_rows_at_loop_start and mtp_rows_at_loop_start)
            if mixed and _stats is not None:
                _stats["mixed_mode_cycles"] += 1

            if scheduler == "adaptive" and mixed and hold_max_steps > 0:
                should_hold = len(mtp_rows_at_loop_start) <= adaptive_hold_mtp_max
                if should_hold and hold_steps < hold_max_steps:
                    if _stats is not None:
                        _stats["adaptive_hold_cycles"] += 1
                        _stats["hold_ar_steps"] += 1
                        _stats["hold_ar_held_mtp_rows"] += len(mtp_rows_at_loop_start)
                    run_ar(ar_rows_at_loop_start, step)
                    hold_steps += 1
                    continue

            if ar_rows_at_loop_start:
                if mixed and _stats is not None:
                    if scheduler == "adaptive":
                        _stats["adaptive_ar_first_cycles"] += 1
                    else:
                        _stats["ar_first_cycles"] += 1
                run_ar(ar_rows_at_loop_start, step)

            mtp_rows = live_rows("mtp")
            if mtp_rows:
                run_mtp(mtp_rows, step)
                hold_steps = 0

            if scheduler == "pipeline" and mtp_rows:
                old_ar = set(ar_rows_at_loop_start)
                new_ar_rows = [b for b in live_rows("ar") if b not in old_ar]
                if new_ar_rows:
                    if _stats is not None:
                        _stats["pipeline_ar_after_mtp_cycles"] += 1
                    run_ar(new_ar_rows, step)
            continue

        mtp_rows = live_rows("mtp")
        ar_rows_at_loop_start = live_rows("ar")
        if mtp_rows and ar_rows_at_loop_start and _stats is not None:
            _stats["mixed_mode_cycles"] += 1
        if mtp_rows:
            run_mtp(mtp_rows, step)

        ar_rows = [b for b in range(bsz) if not finished[b] and modes[b] == "ar"]
        if mtp_rows and ar_rows and _stats is not None:
            _stats["eager_mtp_then_ar_cycles"] += 1
        if ar_rows:
            run_ar(ar_rows, step)

    outs = [
        tok.decode(torch.tensor(gen_ids[b], dtype=torch.long, device=dev),
                   skip_special_tokens=False) if gen_ids[b] else ""
        for b in range(bsz)
    ]
    if top_level_stats:
        if os.environ.get("LA_FLASH_PLAN_STATS", "0") == "1":
            _stats["sparse_plan_stats"] = copy.deepcopy(
                getattr(model, "_la_flash_sparse_plan_stats", None) or {}
            )
        _set_last_hybrid_stats(_stats)
    return outs


@torch.no_grad()
def _step_stock_mtp_rows(model, san, hpat, tids, prompt_ids, kv_rows, rows,
                         cached_lens, full_ids, gen_ids, modes, finished, total_limits,
                         vit_list, pad, mask_tok, img_tok, row_temps, top_p, top_k,
                         repetition_penalty, dev, tok, debug, step_idx, use_magi, stats=None):
    kv, kvalid, old_lens, kmax = _pack_stock_kv_rows(kv_rows, rows, dev)
    uncached_lens = [len(full_ids[r]) - cached_lens[r] for r in rows]
    umax = max(uncached_lens)
    seq_len = umax + N_FUTURE
    _record_forward_stats(stats, "mtp", rows, seq_len, uncached_lens)

    suf_ids = torch.full((len(rows), seq_len), pad, dtype=torch.long, device=dev)
    suf_pos = torch.ones((len(rows), seq_len), dtype=torch.long, device=dev)
    q_valid = torch.zeros((len(rows), seq_len), dtype=torch.long, device=dev)

    for i, r in enumerate(rows):
        uncached = full_ids[r][cached_lens[r] :]
        left = umax - len(uncached)
        if uncached:
            suf_ids[i, left : left + len(uncached)] = torch.tensor(uncached, dtype=torch.long, device=dev)
            suf_pos[i, left : left + len(uncached)] = torch.arange(
                cached_lens[r], len(full_ids[r]), dtype=torch.long, device=dev)
            q_valid[i, left : left + len(uncached)] = 1

        rep = full_ids[r][-1]
        cur_len = len(full_ids[r])
        suf_ids[i, umax] = rep
        suf_pos[i, umax] = cur_len - 1
        q_valid[i, umax] = 1
        for j in range(1, N_FUTURE):
            suf_ids[i, umax + j] = mask_tok
            suf_pos[i, umax + j] = cur_len + (j - 1)
            q_valid[i, umax + j] = 1

    full_mask = torch.cat([kvalid, q_valid], dim=1)

    if debug:
        forward_mask, fallback_rows = _forward_attention_mask(
            model, suf_ids, full_mask, kmax, mtp_window=True, range_plan=True)
        _print_debug_forward(
            f"MTP step={step_idx}",
            model,
            tok,
            suf_ids,
            full_mask,
            suf_pos,
            past_len=kmax,
            mtp_window=True,
            extra={
                "global_rows": rows,
                "old_kv_lens": old_lens,
                "cached_lens": [cached_lens[r] for r in rows],
                "full_lens": [len(full_ids[r]) for r in rows],
                "uncached_lens": uncached_lens,
                "forward_attention_mask": _mask_desc(forward_mask),
                "safe_sdpa_fallback_query_rows": fallback_rows,
            },
            attention_impl="magi" if use_magi else ATTN_MODE,
        )
    else:
        forward_mask, _ = _forward_attention_mask(
            model, suf_ids, full_mask, kmax, mtp_window=True, range_plan=True)
    first_rows = [r for r in rows if cached_lens[r] == 0]
    visual_features = None
    if first_rows:
        if first_rows != rows:
            raise RuntimeError("mixed first/non-first MTP rows are not supported")
        visual_features = torch.cat([vit_list[r] for r in rows], dim=0)
        assert int((suf_ids == img_tok).sum().item()) == visual_features.shape[0], \
            "image-token count != supplied visual_features rows"
    out = language_model_forward(
        model, input_ids=suf_ids, attention_mask=forward_mask,
        position_ids=suf_pos, past_key_values=kv, use_cache=True,
        visual_features=visual_features,
        image_token_index=img_tok if visual_features is not None else None,
        logits_slice=slice(-N_FUTURE, None))

    for i, r in enumerate(rows):
        kv_rows[r] = _unpack_stock_after_forward(
            out.past_key_values, i, old_lens[i], uncached_lens[i], kmax, umax)
        cached_lens[r] = len(full_ids[r])

    wlogits = out.logits[:, -N_FUTURE:, :]
    local_prompts = [prompt_ids[r] for r in rows]
    local_gen = [gen_ids[r] for r in rows]
    gen_pad = _pad_generated(local_prompts, local_gen, img_tok, dev)
    per_row_temp = torch.tensor([row_temps[r] for r in rows], dtype=torch.float32, device=dev)

    if BATCH_SAN:
        x0_all, boxes_all = sample_tokens_batched(
            wlogits, gen_pad, tids, per_row_temp,
            repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
            keep_k_avg=4, generation_mode="hybrid")

    for i, r in enumerate(rows):
        if finished[r]:
            continue
        if BATCH_SAN:
            x0b, boxb = x0_all[i], boxes_all[i]
        else:
            gk = _mk_generate_kwargs(row_temps[r], top_p, top_k, repetition_penalty)
            _, _, x0, box_avg = san(wlogits[i : i + 1], gen_pad[i : i + 1], tids, keep_k=5, **gk)
            x0b, boxb = x0[0], box_avg[0]
        nt = x0b if bool((boxb == 0).all()) else boxb
        op = hpat(nt, tids, "hybrid")

        toks = [int(t) for t in op["tokens"]]
        for t in toks:
            gen_ids[r].append(t)
            full_ids[r].append(t)

        if op["type"] == "im_end":
            finished[r] = True
        elif op["type"] == "error_box":
            modes[r] = "ar"
        if len(full_ids[r]) >= total_limits[r]:
            finished[r] = True


@torch.no_grad()
def _step_stock_ar_rows(model, san, tids, prompt_ids, kv_rows, rows,
                        cached_lens, full_ids, gen_ids, modes, finished, total_limits,
                        pad, img_tok, row_temps, temperature, top_p, top_k,
                        repetition_penalty, dev, tok, debug, step_idx, use_magi, stats=None):
    kv, kvalid, old_lens, kmax = _pack_stock_kv_rows(kv_rows, rows, dev)
    uncached_lens = [len(full_ids[r]) - cached_lens[r] for r in rows]
    if any(n <= 0 for n in uncached_lens):
        raise RuntimeError(f"AR rows have no uncached tokens: {rows}")
    umax = max(uncached_lens)
    _record_forward_stats(stats, "ar", rows, umax, uncached_lens)

    suf_ids = torch.full((len(rows), umax), pad, dtype=torch.long, device=dev)
    suf_pos = torch.ones((len(rows), umax), dtype=torch.long, device=dev)
    q_valid = torch.zeros((len(rows), umax), dtype=torch.long, device=dev)

    for i, r in enumerate(rows):
        uncached = full_ids[r][cached_lens[r] :]
        left = umax - len(uncached)
        suf_ids[i, left:] = torch.tensor(uncached, dtype=torch.long, device=dev)
        suf_pos[i, left:] = torch.arange(cached_lens[r], len(full_ids[r]), dtype=torch.long, device=dev)
        q_valid[i, left:] = 1

    full_mask = torch.cat([kvalid, q_valid], dim=1)
    if debug:
        forward_mask, fallback_rows = _forward_attention_mask(
            model, suf_ids, full_mask, kmax, mtp_window=False, range_plan=True)
        _print_debug_forward(
            f"AR step={step_idx}",
            model,
            tok,
            suf_ids,
            full_mask,
            suf_pos,
            past_len=kmax,
            mtp_window=False,
            extra={
                "global_rows": rows,
                "old_kv_lens": old_lens,
                "cached_lens": [cached_lens[r] for r in rows],
                "full_lens": [len(full_ids[r]) for r in rows],
                "uncached_lens": uncached_lens,
                "forward_attention_mask": _mask_desc(forward_mask),
                "safe_sdpa_fallback_query_rows": fallback_rows,
            },
            attention_impl="magi" if use_magi else ATTN_MODE,
        )
    else:
        forward_mask, _ = _forward_attention_mask(
            model, suf_ids, full_mask, kmax, mtp_window=False, range_plan=True)

    out = language_model_forward(
        model, input_ids=suf_ids, attention_mask=forward_mask,
        position_ids=suf_pos, past_key_values=kv, use_cache=True,
        logits_slice=slice(-1, None))

    for i, r in enumerate(rows):
        kv_rows[r] = _unpack_stock_after_forward(
            out.past_key_values, i, old_lens[i], uncached_lens[i], kmax, umax)
        cached_lens[r] = len(full_ids[r])

    if AR_BATCH_SAN:
        local_prompts = [prompt_ids[r] for r in rows]
        local_gen = [gen_ids[r] for r in rows]
        gen_pad = _pad_generated(local_prompts, local_gen, img_tok, dev)
        per_row_temp = torch.tensor([row_temps[r] for r in rows], dtype=torch.float32, device=dev)
        x0_all = sample_next_tokens_batched(
            out.logits[:, -1:, :],
            gen_pad,
            per_row_temp,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
        )

    for i, r in enumerate(rows):
        if AR_BATCH_SAN:
            token_val = int(x0_all[i, 0].item())
        else:
            logits = out.logits[i : i + 1, -1:, :]
            gen_pad = _pad_generated([prompt_ids[r]], [gen_ids[r]], img_tok, dev)
            gk = _mk_generate_kwargs(temperature, top_p, top_k, repetition_penalty, row_temp=row_temps[r])
            _, _, x0, _ = san(logits, gen_pad, tids, **gk)
            token_val = int(x0[0, 0].item())
        out_type = _classify_ar_token(token_val, tids)

        gen_ids[r].append(token_val)
        full_ids[r].append(token_val)

        if out_type == "im_end":
            finished[r] = True
        elif out_type == "box_end_ar":
            modes[r] = "mtp"

        if len(full_ids[r]) >= total_limits[r]:
            finished[r] = True


def generate_batch_grouped_hybrid(groups, temperature=README_TEMPERATURE, top_p=README_TOP_P,
                                  top_k=None, repetition_penalty=README_REPETITION_PENALTY,
                                  max_new_tokens=README_MAX_NEW_TOKENS, temps=None,
                                  debug=None, scheduler=None, group_size=None,
                                  vision_features=None):
    """Hybrid grouped API shape.

    This preserves grouped return shape, but intentionally uses the generic
    hybrid decoder rather than the fast engine's shared-prefix optimization.
    """
    flat = []
    flat_vision_features = [] if vision_features is not None else None
    counts = []
    for group_idx, (im, queries) in enumerate(groups):
        counts.append(len(queries))
        flat.extend((im, q) for q in queries)
        if flat_vision_features is not None:
            flat_vision_features.extend([vision_features[group_idx]] * len(queries))

    outs = generate_batch_hybrid(
        flat, temperature=temperature, top_p=top_p, top_k=top_k,
        repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        temps=temps, debug=debug, scheduler=scheduler, group_size=group_size,
        vision_features=flat_vision_features)
    res, offset = [], 0
    for n in counts:
        res.append(outs[offset : offset + n])
        offset += n
    return res


__all__ = ["generate_batch_hybrid", "generate_batch_grouped_hybrid", "get_last_hybrid_stats"]
