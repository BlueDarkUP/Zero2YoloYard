"""Internal runtime support for the LocateAnything-3B hybrid batch decoder.

This file keeps only the model-loading, tokenization, image-encoding, stock
processor, and sample-token helpers that ``engine_hybrid.py`` needs.

Important env knobs:
  LA_FLASH_MODEL          HF repo id / local path of the model (default nvidia/LocateAnything-3B)
  HF_HUB_OFFLINE=1    read the local HF cache only (no network); unset -> download on first use
  LA_FLASH_ATTN           sdpa, eager, magi, or la_flash; la_flash uses FlashAttention sparse ranges
  LA_FLASH_STRICT_ATTN    1 -> fail if the requested backend is unavailable;
                      default 0 falls back to sdpa
  LA_FLASH_VISION_ATTN    auto, flash_attention_2, sdpa, or eager (default auto)
  LA_FLASH_HYBRID_PREFILL shared, none, per_row, or batch prompt KV prefill (default shared)
  MTP_BATCH_VISION    0 -> per-image vision encode (default 1: batched when flash is present)
  LA_FLASH_VISION_ENCODE_BATCH_SIZE
                      max images per MoonViT encode micro-batch (default 8; <=0 disables limit)
  MTP_BATCH_SAN       0 -> per-row logits/sample pipeline (default 1: batched over [B,6,V])
  AR_BATCH_SAN        0 -> per-row AR sample pipeline (default 1: batched over [B,1,V])
"""
import inspect
import os, warnings, importlib, torch
from types import SimpleNamespace
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor


# By default let transformers fetch the model on first use; set HF_HUB_OFFLINE=1 yourself
# to read the local HF cache only (e.g. air-gapped / already-downloaded runs).
MODEL = os.environ.get("LA_FLASH_MODEL", "nvidia/LocateAnything-3B")


LLM_ATTN_MODES = ("sdpa", "eager", "magi", "la_flash")
VISION_ATTN_MODES = ("auto", "flash_attention_2", "sdpa", "eager")


def _normalize_attn_mode(value):
    mode = (value or "sdpa").strip().lower().replace("-", "_")
    aliases = {
        "": "sdpa",
        "manual": "eager",
        "torch": "eager",
        "torch_eager": "eager",
        "torch_sdpa": "sdpa",
        "scaled_dot_product_attention": "sdpa",
        "flash": "la_flash",
        "la_flash": "la_flash",
        "kernel": "la_flash",
        "cuda": "la_flash",
        "range": "la_flash",
        "range_attention": "la_flash",
        "flex_flash": "magi",
        "flex_flash_attention": "magi",
        "flex_flash_attn": "magi",
    }
    mode = aliases.get(mode, mode)
    if mode not in LLM_ATTN_MODES:
        raise ValueError(
            f"LA_FLASH_ATTN must be one of {', '.join(LLM_ATTN_MODES)}; got {value!r}"
        )
    return mode


def _normalize_vision_attn_mode(value):
    mode = (value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "flash": "flash_attention_2",
        "flash_attention2": "flash_attention_2",
        "fa2": "flash_attention_2",
        "manual": "eager",
    }
    mode = aliases.get(mode, mode)
    if mode not in VISION_ATTN_MODES:
        raise ValueError(
            f"LA_FLASH_VISION_ATTN must be one of {', '.join(VISION_ATTN_MODES)}; got {value!r}"
        )
    return mode


ATTN_MODE = _normalize_attn_mode(os.environ.get("LA_FLASH_ATTN", "sdpa"))
REMOTE_ATTN_MODE = "sdpa" if ATTN_MODE in {"la_flash", "magi"} else ATTN_MODE
VISION_ATTN_MODE = _normalize_vision_attn_mode(os.environ.get("LA_FLASH_VISION_ATTN", "auto"))
MAX_DIM = 1024
DEV, DT = "cuda", torch.bfloat16
N_FUTURE = 6                                            # = config.block_size (MTP window)
_PROMPT = "Locate all the instances that matches the following description: "


def _env_flag(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return None
    return int(val)


def _strict_attn():
    return _env_flag("LA_FLASH_STRICT_ATTN", False)


def _fallback_to_sdpa(model, requested, reason):
    if requested == "sdpa":
        raise RuntimeError(f"LA_FLASH_ATTN=sdpa failed: {reason}") from reason
    message = f"LA_FLASH_ATTN={requested} is unavailable; falling back to sdpa. Reason: {reason}"
    if _strict_attn():
        raise RuntimeError(message) from reason
    warnings.warn(message)
    _set_llm_mode(model, "sdpa")
    model._la_flash_requested_attn_original = requested
    model._la_flash_attn_fallback_reason = str(reason)
    return "sdpa"


# Optional compile for the shared Qwen2 core. This is off by default because the
# hybrid scheduler already varies query/cache shapes and first-call compile cost is high.
MTP_COMPILE = os.environ.get("MTP_COMPILE", "0") == "1"

# Batch the MoonViT vision encode across a micro-batch's images: pack N images into ONE
# extract_feature. With flash present, MoonViT's varlen cu_seqlens path is block-diagonal per
# image and equivalent to per-image encode.
# Without flash, sdpa builds a dense [1,S,S] mask -> O(S^2) N^2 -> per-image fallback (auto, see
# _vision_is_flash). Default ON; set MTP_BATCH_VISION=0 to force per-image.
BATCH_VISION = os.environ.get("MTP_BATCH_VISION", "1") == "1"
_vision_encode_batch_size = _env_int("LA_FLASH_VISION_ENCODE_BATCH_SIZE")
VISION_ENCODE_BATCH_SIZE = 8 if _vision_encode_batch_size is None else max(0, _vision_encode_batch_size)

# Batch the per-row box-decode (sample_tokens): run the row-independent logits pipeline
# (rep-penalty / per-row temperature / top_p / top_k / softmax / sample) ONCE over the whole
# [B,6,V] step instead of B times on [1,6,V]; only the variable-length box assembly stays per-row.
# Greedy is BIT-IDENTICAL to the per-row san (argmax, no RNG). Default ON; MTP_BATCH_SAN=0 -> per-row.
BATCH_SAN = os.environ.get("MTP_BATCH_SAN", "1") == "1"

# Batch the AR repair sampler over [B,1,V]. This shares the exact filtering
# helpers with MTP batching but skips box/ref decoding, so it only replaces the
# repeated stock one-token sample calls.  Sampling itself stays row-ordered by
# default to preserve the stock RNG consumption pattern for AR repair.
AR_BATCH_SAN = os.environ.get("AR_BATCH_SAN", "1") == "1"

_tok = _proc = _model = None

def _magi_diag():
    lines = []
    try:
        import magi_attention
        lines.append(f"magi_attention: OK file={getattr(magi_attention, '__file__', None)}")
        lines.append(f"magi_attention.__version__={getattr(magi_attention, '__version__', '<missing>')}")
    except Exception as e:
        lines.append(f"magi_attention: FAIL {type(e).__name__}: {e}")
        return "\n".join(lines)
    try:
        from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
        lines.append(f"magi_attention.functional.flex_flash_attn: OK func={flex_flash_attn_func}")
    except Exception as e:
        lines.append(f"magi_attention.functional.flex_flash_attn: FAIL {type(e).__name__}: {e}")
    return "\n".join(lines)

def _remote_magi_diag(model=None):
    lines = []
    try:
        if model is not None:
            mod = importlib.import_module(type(model.language_model.model).__module__)
        else:
            # Best effort: if the dynamic module is not imported yet this may fail;
            # the post-load diagnostic below will still work.
            mod = importlib.import_module("transformers_modules.LocateAnything-3B.modeling_qwen2")
        lines.append(f"remote_qwen2_module={getattr(mod, '__file__', None)}")
        lines.append(f"remote_qwen2._MAGI_AVAILABLE={getattr(mod, '_MAGI_AVAILABLE', '<missing>')!r}")
        lines.append(f"remote_qwen2.flex_flash_attn_func={getattr(mod, 'flex_flash_attn_func', '<missing>')}")
    except Exception as e:
        lines.append(f"remote_qwen2: diagnostic failed {type(e).__name__}: {e}")
    return "\n".join(lines)

def _attn_class_diag(model):
    try:
        llm = model.language_model.model
        classes = [type(layer.self_attn).__name__ for layer in llm.layers[:4]]
        return (
            f"llm._attn_implementation={getattr(llm, '_attn_implementation', None)!r}\n"
            f"config._attn_implementation={getattr(llm.config, '_attn_implementation', None)!r}\n"
            f"first_attn_classes={classes}"
        )
    except Exception as e:
        return f"attention class diagnostic failed {type(e).__name__}: {e}"


def _set_vision_attention_mode(model):
    """Match HF's MoonViT policy: prefer flash_attention_2, then sdpa, then eager."""
    vm = getattr(model, "vision_model", None)
    if vm is None:
        return None
    mod = importlib.import_module(type(vm).__module__)
    funcs = getattr(mod, "VL_VISION_ATTENTION_FUNCTIONS", {})
    has_flash = getattr(mod, "flash_attn_varlen_func", None) is not None
    requested = VISION_ATTN_MODE

    if requested == "auto":
        candidates = ("flash_attention_2", "sdpa", "eager")
    else:
        candidates = (requested, "flash_attention_2", "sdpa", "eager")

    chosen = None
    for candidate in candidates:
        if candidate == "flash_attention_2" and not has_flash:
            continue
        if candidate in funcs:
            chosen = candidate
            break
    if chosen is None:
        raise RuntimeError("MoonViT has no supported attention implementation.")

    if requested == "flash_attention_2" and chosen != "flash_attention_2":
        warnings.warn("LA_FLASH_VISION_ATTN=flash_attention_2 requested but flash-attn is unavailable; "
                      f"using {chosen}.")
    elif requested not in {"auto", chosen}:
        warnings.warn(f"LA_FLASH_VISION_ATTN={requested} is unavailable; using {chosen}.")

    if hasattr(model.config, "vision_config"):
        model.config.vision_config._attn_implementation = chosen
    try:
        vm.config._attn_implementation = chosen
    except Exception:
        pass
    try:
        for block in vm.encoder.blocks:
            block.attn_implementation = chosen
    except Exception as exc:
        raise RuntimeError("Failed to configure MoonViT attention implementation.") from exc
    model._la_flash_vision_attn = chosen
    return chosen


def load():
    """Lazy model load with HF remote-code semantics plus release backends.

    The text decoder is pinned to one of sdpa/eager/magi/la_flash. MoonViT is
    configured independently and follows the HF policy: flash_attention_2 when
    flash-attn is importable, otherwise sdpa, otherwise eager.
    """
    global _tok, _proc, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        _proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        attn_impl = REMOTE_ATTN_MODE
        if ATTN_MODE == "magi" and os.environ.get("LA_FLASH_DEBUG", "0") != "0":
            print("LA Flash magi pre-load diagnostic:", flush=True)
            print(_magi_diag(), flush=True)
        _model = AutoModel.from_pretrained(MODEL, torch_dtype=DT, trust_remote_code=True,
                                           attn_implementation=attn_impl).to(DEV).eval()
        _set_vision_attention_mode(_model)
        actual_attn = getattr(_model.language_model.model, "_attn_implementation", None)
        if ATTN_MODE == "magi" and os.environ.get("LA_FLASH_DEBUG", "0") != "0":
            print("LA Flash magi post-load diagnostic:", flush=True)
            print(_remote_magi_diag(_model), flush=True)
            print(_attn_class_diag(_model), flush=True)
        if ATTN_MODE == "magi":
            try:
                qwen2_mod = importlib.import_module(type(_model.language_model.model).__module__)
                if not getattr(qwen2_mod, "_MAGI_AVAILABLE", False):
                    raise RuntimeError(
                        "remote module reports _MAGI_AVAILABLE=False.\n"
                        f"{_remote_magi_diag(_model)}\n{_magi_diag()}"
                    )
                first_attn = type(_model.language_model.model.layers[0].self_attn).__name__
                if actual_attn != "sdpa" or first_attn != "_BatchedMagiAttention":
                    _set_llm_mode(_model, "magi")
                    actual_attn = getattr(_model.language_model.model, "_attn_implementation", None)
                    first_attn = type(_model.language_model.model.layers[0].self_attn).__name__
                    if os.environ.get("LA_FLASH_DEBUG", "0") != "0":
                        print("LA Flash magi post-swap diagnostic:", flush=True)
                        print(_attn_class_diag(_model), flush=True)
                if actual_attn != "sdpa" or first_attn != "_BatchedMagiAttention":
                    raise RuntimeError(
                        "batched magi attention did not activate. "
                        f"actual_attn={actual_attn!r}; first_attn={first_attn!r}; "
                        f"{_remote_magi_diag(_model)}; {_attn_class_diag(_model)}"
                    )
                _model._la_flash_requested_attn = "magi"
            except Exception as exc:
                _fallback_to_sdpa(_model, "magi", exc)
        else:
            try:
                _set_llm_mode(_model, ATTN_MODE)   # decode-safe mask plumbing for sdpa/eager/la_flash
            except Exception as exc:
                _fallback_to_sdpa(_model, ATTN_MODE, exc)
        if MTP_COMPILE:
            _maybe_compile(_model)
    return _tok, _proc, _model


def _maybe_compile(model):
    """Compile the shared Qwen2Model core (base.forward). It backs BOTH prefill (called directly)
    and decode (language_model.forward -> self.model). lm_head + MoonViT left eager. dynamic=True
    so the varying decode S/kvlen don't trigger a recompile storm. No-op + warning if triton is
    missing (inductor needs it on GPU). First call pays the compile cost (~42s warm / ~187s cold)."""
    try:
        import triton  # noqa: F401
    except Exception:
        warnings.warn("MTP_COMPILE set but triton is unavailable; running without torch.compile.")
        return
    import torch._dynamo as _dyn
    _dyn.config.cache_size_limit = max(_dyn.config.cache_size_limit, 64)
    base = model.language_model.model
    if not getattr(base, "_mtp_compiled", False):
        base.forward = torch.compile(base.forward, dynamic=True)
        base._mtp_compiled = True


def build_batched_magi_attention_class(mod):
    """Build a Qwen2 attention subclass backed by Magi's flex_flash_attn.

    The official LocateAnything ``Qwen2MagiAttention`` asserts ``bsz == 1`` and
    relies on ``Qwen2Model._attn_implementation == "magi"`` to build a single
    sample range plan.  For release batch inference the hybrid scheduler passes
    a batched Magi range plan directly to this layer; a 4D-mask conversion path
    remains as a compatibility fallback.
    """
    flex_flash_attn_func = getattr(mod, "flex_flash_attn_func", None)
    if flex_flash_attn_func is None:
        try:
            from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
        except Exception as exc:
            raise RuntimeError(
                "LA_FLASH_ATTN=magi requires "
                "magi_attention.functional.flex_flash_attn.flex_flash_attn_func."
            ) from exc

    FULL, CAUSAL = 0, 1
    causal_plan_cache = {}
    try:
        magi_params = set(inspect.signature(flex_flash_attn_func).parameters)
    except (TypeError, ValueError):
        magi_params = set()
    supports_disable_fwd_atomic = "disable_fwd_atomic_reduction" in magi_params

    def _disjoint_q_ranges(q_ranges):
        seen = set()
        for start, end in q_ranges:
            key = (int(start), int(end))
            if key in seen:
                return False
            seen.add(key)
        return True

    def _plan_disjoint_q_ranges(plan):
        cached = plan.get("_la_flash_disjoint_q_ranges")
        if cached is not None:
            return bool(cached)
        q_ranges = plan["q_ranges"].detach().to(device="cpu", dtype=torch.int32).tolist()
        disjoint = _disjoint_q_ranges(q_ranges)
        try:
            plan["_la_flash_disjoint_q_ranges"] = disjoint
        except Exception:
            pass
        return disjoint

    def _tensor_plan(q_ranges, k_ranges, types, device):
        return {
            "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
            "_la_flash_disjoint_q_ranges": _disjoint_q_ranges(q_ranges),
        }

    def _offset_plan(plan, q_offset, k_offset):
        return (
            (plan["q_ranges"] + int(q_offset)).tolist(),
            (plan["k_ranges"] + int(k_offset)).tolist(),
            plan["attn_type_map"].tolist(),
        )

    def _causal_plan(bsz, q_len, kv_seq_len, device):
        key = (int(bsz), int(q_len), int(kv_seq_len), device.type, device.index)
        cached = causal_plan_cache.get(key)
        if cached is not None:
            return cached
        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            q_base = b * int(q_len)
            k_base = b * int(kv_seq_len)
            q_ranges.append([q_base, q_base + int(q_len)])
            k_ranges.append([k_base, k_base + int(kv_seq_len)])
            types.append(CAUSAL)
        plan = _tensor_plan(q_ranges, k_ranges, types, device)
        plan.update(
            {
                "flash_cu_seqlens_q": torch.arange(
                    0,
                    (int(bsz) + 1) * int(q_len),
                    int(q_len),
                    dtype=torch.int32,
                    device=device,
                ),
                "flash_cu_seqlens_k": torch.arange(
                    0,
                    (int(bsz) + 1) * int(kv_seq_len),
                    int(kv_seq_len),
                    dtype=torch.int32,
                    device=device,
                ),
                "flash_causal": True,
            }
        )
        causal_plan_cache[key] = plan
        return plan

    def _row_segments(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return ((0, 1),)
        split = np.flatnonzero(np.diff(idx) > 1) + 1
        starts = np.concatenate((idx[:1], idx[split]))
        ends = np.concatenate((idx[split - 1], idx[-1:])) + 1
        return tuple((int(s), int(e)) for s, e in zip(starts, ends))

    def _visible_from_4d_mask(attention_mask, kv_seq_len):
        mask = attention_mask[:, :, :, :kv_seq_len]
        if mask.dtype == torch.bool:
            return mask[:, 0].detach().to(device="cpu", dtype=torch.bool).contiguous()
        mask_cpu = mask[:, 0].detach().to(device="cpu").contiguous()
        if getattr(attention_mask, "_la_flash_visible_mask", False):
            return (mask_cpu > 0).to(dtype=torch.bool)

        max_value = float(mask_cpu.max().item()) if mask_cpu.numel() else 0.0
        min_value = float(mask_cpu.min().item()) if mask_cpu.numel() else 0.0
        if max_value > 0.0 and min_value >= 0.0:
            return (mask_cpu > 0).to(dtype=torch.bool)
        return (mask_cpu >= 0).to(dtype=torch.bool)

    def _plan_from_visible_mask(attention_mask, bsz, q_len, kv_seq_len, device):
        cache_key = (int(bsz), int(q_len), int(kv_seq_len), device.type, device.index)
        cached = getattr(attention_mask, "_la_flash_magi_plan", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        visible = _visible_from_4d_mask(attention_mask, int(kv_seq_len)).numpy()
        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            q_base = b * int(q_len)
            k_base = b * int(kv_seq_len)
            run_start = 0
            run_segments = _row_segments(visible[b, 0])
            for q in range(1, int(q_len)):
                segments = _row_segments(visible[b, q])
                if segments == run_segments:
                    continue
                for start, end in run_segments:
                    q_ranges.append([q_base + run_start, q_base + q])
                    k_ranges.append([k_base + start, k_base + end])
                    types.append(FULL)
                run_start = q
                run_segments = segments
            for start, end in run_segments:
                q_ranges.append([q_base + run_start, q_base + int(q_len)])
                k_ranges.append([k_base + start, k_base + end])
                types.append(FULL)

        plan = _tensor_plan(q_ranges, k_ranges, types, device)
        try:
            attention_mask._la_flash_magi_plan = (cache_key, plan)
        except Exception:
            pass
        return plan

    def _plan_from_magi_dict(attention_mask, bsz, q_len, kv_seq_len, device):
        if int(bsz) == 1:
            return attention_mask
        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            qs, ks, ts = _offset_plan(
                attention_mask,
                q_offset=b * int(q_len),
                k_offset=b * int(kv_seq_len),
            )
            q_ranges.extend(qs)
            k_ranges.extend(ks)
            types.extend(ts)
        return _tensor_plan(q_ranges, k_ranges, types, device)

    def _magi_plan(attention_mask, bsz, q_len, kv_seq_len, device):
        if isinstance(attention_mask, dict):
            if attention_mask.get("_la_flash_batched", False):
                return attention_mask
            return _plan_from_magi_dict(attention_mask, bsz, q_len, kv_seq_len, device)
        if attention_mask is None:
            return _causal_plan(bsz, q_len, kv_seq_len, device)
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )
        return _plan_from_visible_mask(attention_mask, bsz, q_len, kv_seq_len, device)

    class _BatchedMagiAttention(mod.Qwen2Attention):
        """MagiAttention path with true batch inference via packed token ranges."""

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            if output_attentions:
                raise NotImplementedError("MagiAttention does not support output_attentions=True")

            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using "
                        f"{self.__class__.__name__} for auto-regressive decoding with k/v caching, "
                        "please initialize the attention class with a layer index."
                    )
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = mod.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs)

            kv_seq_len = key_states.shape[-2]
            plan = _magi_plan(attention_mask, bsz, q_len, kv_seq_len, query_states.device)
            magi_extra_kwargs = {}
            if supports_disable_fwd_atomic:
                magi_extra_kwargs["disable_fwd_atomic_reduction"] = (
                    (not self.training) and _plan_disjoint_q_ranges(plan)
                )

            query_states = query_states.transpose(1, 2).reshape(
                bsz * q_len, self.num_heads, self.head_dim).contiguous()
            key_states = key_states.transpose(1, 2).reshape(
                bsz * kv_seq_len, self.num_key_value_heads, self.head_dim).contiguous()
            value_states = value_states.transpose(1, 2).reshape(
                bsz * kv_seq_len, self.num_key_value_heads, self.head_dim).contiguous()

            attn_output, _ = flex_flash_attn_func(
                query_states,
                key_states,
                value_states,
                q_ranges=plan["q_ranges"],
                k_ranges=plan["k_ranges"],
                attn_type_map=plan["attn_type_map"],
                softmax_scale=getattr(self, "softmax_scale", self.head_dim ** -0.5),
                softcap=0.0,
                deterministic=False,
                **magi_extra_kwargs,
            )
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

    return _BatchedMagiAttention


def build_la_flash_attention_class(mod):
    """Build a Qwen2 attention subclass backed by LA Flash sparse ranges."""
    try:
        from kernel_utils import is_available, range_attention
    except Exception as exc:
        raise RuntimeError(
            "LA_FLASH_ATTN=la_flash requires kernel_utils and FlashAttention."
        ) from exc
    if not is_available():
        raise RuntimeError(
            "LA_FLASH_ATTN=la_flash requires flash_attn.flash_attn_varlen_func."
        )

    FULL, CAUSAL = 0, 1
    causal_plan_cache = {}

    def _tensor_plan(q_ranges, k_ranges, types, device):
        max_q_len = max((int(end) - int(start) for start, end in q_ranges), default=0)
        max_k_len = max((int(end) - int(start) for start, end in k_ranges), default=0)
        plan = {
            "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=device).contiguous(),
            "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=device).contiguous(),
            "attn_type_map": torch.tensor(types, dtype=torch.int32, device=device).contiguous(),
            "max_q_len": max_q_len,
            "max_k_len": max_k_len,
        }
        plan.update(_la_flash_group_plan_tensors(q_ranges, types, device))
        return plan

    def _offset_plan(plan, q_offset, k_offset):
        return (
            (plan["q_ranges"] + int(q_offset)).tolist(),
            (plan["k_ranges"] + int(k_offset)).tolist(),
            plan["attn_type_map"].tolist(),
        )

    def _causal_plan(bsz, q_len, kv_seq_len, device):
        key = (int(bsz), int(q_len), int(kv_seq_len), device.type, device.index)
        cached = causal_plan_cache.get(key)
        if cached is not None:
            return cached
        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            q_base = b * int(q_len)
            k_base = b * int(kv_seq_len)
            q_ranges.append([q_base, q_base + int(q_len)])
            k_ranges.append([k_base, k_base + int(kv_seq_len)])
            types.append(CAUSAL)
        plan = _tensor_plan(q_ranges, k_ranges, types, device)
        plan.update(
            {
                "flash_cu_seqlens_q": torch.arange(
                    0,
                    (int(bsz) + 1) * int(q_len),
                    int(q_len),
                    dtype=torch.int32,
                    device=device,
                ),
                "flash_cu_seqlens_k": torch.arange(
                    0,
                    (int(bsz) + 1) * int(kv_seq_len),
                    int(kv_seq_len),
                    dtype=torch.int32,
                    device=device,
                ),
                "flash_causal": True,
            }
        )
        causal_plan_cache[key] = plan
        return plan

    def _row_segments(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return ((0, 1),)
        split = np.flatnonzero(np.diff(idx) > 1) + 1
        starts = np.concatenate((idx[:1], idx[split]))
        ends = np.concatenate((idx[split - 1], idx[-1:])) + 1
        return tuple((int(s), int(e)) for s, e in zip(starts, ends))

    def _visible_from_4d_mask(attention_mask, kv_seq_len):
        mask = attention_mask[:, :, :, :kv_seq_len]
        if mask.dtype == torch.bool:
            return mask[:, 0].detach().to(device="cpu", dtype=torch.bool).contiguous()
        mask_cpu = mask[:, 0].detach().to(device="cpu").contiguous()
        if getattr(attention_mask, "_la_flash_visible_mask", False):
            return (mask_cpu > 0).to(dtype=torch.bool)

        max_value = float(mask_cpu.max().item()) if mask_cpu.numel() else 0.0
        min_value = float(mask_cpu.min().item()) if mask_cpu.numel() else 0.0
        if max_value > 0.0 and min_value >= 0.0:
            return (mask_cpu > 0).to(dtype=torch.bool)
        return (mask_cpu >= 0).to(dtype=torch.bool)

    def _prefix_len(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return None
        end = int(idx[-1]) + 1
        if not bool(row[:end].all()) or bool(row[end:].any()):
            return None
        return end

    def _causal_plan_from_visible(visible, bsz, q_len, kv_seq_len, device):
        q_ranges, k_ranges, types = [], [], []
        packed_flash = True
        for b in range(int(bsz)):
            first_len = _prefix_len(visible[b, 0])
            if first_len is None:
                return None
            valid_len = int(first_len) + int(q_len) - 1
            if valid_len < int(q_len) or valid_len > int(kv_seq_len):
                return None
            for q in range(int(q_len)):
                row_len = _prefix_len(visible[b, q])
                expected = valid_len - int(q_len) + q + 1
                if row_len != expected:
                    return None
            q_base = b * int(q_len)
            k_base = b * int(kv_seq_len)
            q_ranges.append([q_base, q_base + int(q_len)])
            k_ranges.append([k_base, k_base + valid_len])
            types.append(CAUSAL)
            packed_flash = packed_flash and valid_len == int(kv_seq_len)

        plan = _tensor_plan(q_ranges, k_ranges, types, device)
        plan["_la_flash_disjoint_q_ranges"] = True
        if packed_flash:
            plan.update(
                {
                    "flash_cu_seqlens_q": torch.arange(
                        0,
                        (int(bsz) + 1) * int(q_len),
                        int(q_len),
                        dtype=torch.int32,
                        device=device,
                    ),
                    "flash_cu_seqlens_k": torch.arange(
                        0,
                        (int(bsz) + 1) * int(kv_seq_len),
                        int(kv_seq_len),
                        dtype=torch.int32,
                        device=device,
                    ),
                    "flash_causal": True,
                }
            )
        return plan

    def _plan_from_visible_mask(attention_mask, bsz, q_len, kv_seq_len, device):
        cache_key = (int(bsz), int(q_len), int(kv_seq_len), device.type, device.index, "la_flash")
        cached = getattr(attention_mask, "_la_flash_range_plan", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        visible = _visible_from_4d_mask(attention_mask, int(kv_seq_len)).numpy()
        plan = _causal_plan_from_visible(visible, bsz, q_len, kv_seq_len, device)
        if plan is not None:
            try:
                attention_mask._la_flash_range_plan = (cache_key, plan)
            except Exception:
                pass
            return plan

        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            q_base = b * int(q_len)
            k_base = b * int(kv_seq_len)
            run_start = 0
            run_segments = _row_segments(visible[b, 0])
            for q in range(1, int(q_len)):
                segments = _row_segments(visible[b, q])
                if segments == run_segments:
                    continue
                for start, end in run_segments:
                    q_ranges.append([q_base + run_start, q_base + q])
                    k_ranges.append([k_base + start, k_base + end])
                    types.append(FULL)
                run_start = q
                run_segments = segments
            for start, end in run_segments:
                q_ranges.append([q_base + run_start, q_base + int(q_len)])
                k_ranges.append([k_base + start, k_base + end])
                types.append(FULL)

        plan = _tensor_plan(q_ranges, k_ranges, types, device)
        try:
            attention_mask._la_flash_range_plan = (cache_key, plan)
        except Exception:
            pass
        return plan

    def _plan_from_magi_dict(attention_mask, bsz, q_len, kv_seq_len, device):
        if int(bsz) == 1:
            return attention_mask
        q_ranges, k_ranges, types = [], [], []
        for b in range(int(bsz)):
            qs, ks, ts = _offset_plan(
                attention_mask,
                q_offset=b * int(q_len),
                k_offset=b * int(kv_seq_len),
            )
            q_ranges.extend(qs)
            k_ranges.extend(ks)
            types.extend(ts)
        return _tensor_plan(q_ranges, k_ranges, types, device)

    def _range_plan(attention_mask, bsz, q_len, kv_seq_len, device):
        if isinstance(attention_mask, dict):
            if attention_mask.get("_la_flash_batched", False):
                return attention_mask
            return _plan_from_magi_dict(attention_mask, bsz, q_len, kv_seq_len, device)
        if attention_mask is None:
            return _causal_plan(bsz, q_len, kv_seq_len, device)
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                f"but is {attention_mask.size()}"
            )
        return _plan_from_visible_mask(attention_mask, bsz, q_len, kv_seq_len, device)

    class _LaFlashAttention(mod.Qwen2Attention):
        """Range-plan attention path backed by FlashAttention sparse ranges."""

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            if output_attentions:
                raise NotImplementedError("LA Flash attention does not support output_attentions=True")

            bsz, q_len, _ = hidden_states.size()
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using "
                        f"{self.__class__.__name__} for auto-regressive decoding with k/v caching, "
                        "please initialize the attention class with a layer index."
                    )
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = mod.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs)

            kv_seq_len = key_states.shape[-2]
            dense_backend = os.environ.get("LA_FLASH_DENSE_BACKEND", "sdpa").strip().lower()
            if dense_backend == "sdpa" and not isinstance(attention_mask, dict):
                dense_key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
                dense_value_states = mod.repeat_kv(value_states, self.num_key_value_groups)
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        raise ValueError(
                            f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, "
                            f"but is {attention_mask.size()}"
                        )
                    query_for_sdpa = query_states.contiguous()
                    key_for_sdpa = dense_key_states.contiguous()
                    value_for_sdpa = dense_value_states.contiguous()
                    is_causal = False
                elif past_key_value is None:
                    query_for_sdpa = query_states
                    key_for_sdpa = dense_key_states
                    value_for_sdpa = dense_value_states
                    is_causal = bool(self.is_causal and q_len > 1)
                else:
                    query_for_sdpa = key_for_sdpa = value_for_sdpa = None
                    is_causal = False
                if query_for_sdpa is not None:
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        query_for_sdpa,
                        key_for_sdpa,
                        value_for_sdpa,
                        attn_mask=attention_mask,
                        dropout_p=self.attention_dropout if self.training else 0.0,
                        is_causal=is_causal,
                    )
                    attn_output = attn_output.transpose(1, 2).contiguous()
                    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
                    attn_output = self.o_proj(attn_output)
                    return attn_output, None, past_key_value

            plan = _range_plan(attention_mask, bsz, q_len, kv_seq_len, query_states.device)

            query_states = query_states.transpose(1, 2).reshape(
                bsz * q_len, self.num_heads, self.head_dim).contiguous()
            key_states = key_states.transpose(1, 2).reshape(
                bsz * kv_seq_len, self.num_key_value_heads, self.head_dim).contiguous()
            value_states = value_states.transpose(1, 2).reshape(
                bsz * kv_seq_len, self.num_key_value_heads, self.head_dim).contiguous()

            attn_output = range_attention(
                query_states,
                key_states,
                value_states,
                plan["q_ranges"],
                plan["k_ranges"],
                plan["attn_type_map"],
                getattr(self, "softmax_scale", self.head_dim ** -0.5),
                segment_offsets=plan.get("segment_offsets"),
                group_q_ranges=plan.get("group_q_ranges"),
                group_attn_type_map=plan.get("group_attn_type_map"),
                max_q_len=plan.get("max_q_len"),
                max_k_len=plan.get("max_k_len"),
                flash_cu_seqlens_q=plan.get("flash_cu_seqlens_q"),
                flash_cu_seqlens_k=plan.get("flash_cu_seqlens_k"),
                flash_causal=plan.get("flash_causal"),
                disjoint_q_ranges=plan.get("_la_flash_disjoint_q_ranges"),
            )
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

    return _LaFlashAttention


def _is_magi_plan(obj):
    return isinstance(obj, dict) and {
        "q_ranges",
        "k_ranges",
        "attn_type_map",
    }.issubset(obj.keys())


def _la_flash_group_plan_tensors(q_ranges, types, device):
    """Group consecutive Magi range entries that share the same query span.

    Magi-style plans may represent one query span with multiple disjoint key
    spans.  LA Flash consumes those as one FlashAttention-backed softmax group.
    """
    if not q_ranges:
        return {
            "group_q_ranges": torch.empty((0, 2), dtype=torch.int32, device=device),
            "segment_offsets": torch.zeros((1,), dtype=torch.int32, device=device),
            "group_attn_type_map": torch.empty((0,), dtype=torch.int32, device=device),
        }

    grouped_q, grouped_types, offsets = [], [], [0]
    last_q = None
    last_type = None
    for idx, (q_range, attn_type) in enumerate(zip(q_ranges, types)):
        key = (int(q_range[0]), int(q_range[1]))
        attn_type = int(attn_type)
        if last_q is None:
            grouped_q.append([key[0], key[1]])
            grouped_types.append(attn_type)
            last_q = key
            last_type = attn_type
            continue
        if key == last_q and attn_type == last_type:
            continue
        offsets.append(idx)
        grouped_q.append([key[0], key[1]])
        grouped_types.append(attn_type)
        last_q = key
        last_type = attn_type
    offsets.append(len(q_ranges))

    return {
        "group_q_ranges": torch.tensor(grouped_q, dtype=torch.int32, device=device).contiguous(),
        "segment_offsets": torch.tensor(offsets, dtype=torch.int32, device=device).contiguous(),
        "group_attn_type_map": torch.tensor(grouped_types, dtype=torch.int32, device=device).contiguous(),
        "max_q_len": max((end - start for start, end in grouped_q), default=0),
    }


def _record_sparse_plan_stats(model, q_ranges, k_ranges, types):
    if os.environ.get("LA_FLASH_PLAN_STATS", "0") != "1":
        return
    stats = getattr(model, "_la_flash_sparse_plan_stats", None)
    if stats is None:
        stats = {
            "calls": 0,
            "ranges": 0,
            "q_tokens": 0,
            "k_tokens": 0,
            "max_q_len": 0,
            "max_k_len": 0,
            "full_ranges": 0,
            "causal_ranges": 0,
            "other_ranges": 0,
        }
        model._la_flash_sparse_plan_stats = stats
    stats["calls"] += 1
    stats["ranges"] += len(q_ranges)
    for (q_start, q_end), (k_start, k_end), attn_type in zip(q_ranges, k_ranges, types):
        q_len = int(q_end) - int(q_start)
        k_len = int(k_end) - int(k_start)
        stats["q_tokens"] += q_len
        stats["k_tokens"] += k_len
        stats["max_q_len"] = max(stats["max_q_len"], q_len)
        stats["max_k_len"] = max(stats["max_k_len"], k_len)
        attn_type = int(attn_type)
        if attn_type == 0:
            stats["full_ranges"] += 1
        elif attn_type == 1:
            stats["causal_ranges"] += 1
        else:
            stats["other_ranges"] += 1


def build_magi_scheduler_ranges(model, attention_mask_2d, input_ids, past_len, mtp_window=False):
    """Build batched Magi ranges directly from the hybrid scheduler mask.

    The official Qwen2 SDPA dispatcher may optimize an all-valid 2D mask to
    ``None`` before decoder layers see it. That is correct for plain causal
    attention but loses LocateAnything's MTP generation-window rule. Building
    ranges here keeps Magi batch inference exact and avoids per-layer dense
    mask conversion.
    """
    requested_attn = getattr(model, "_la_flash_requested_attn", ATTN_MODE)
    if requested_attn not in {"magi", "la_flash"}:
        return None
    if attention_mask_2d is None or not hasattr(attention_mask_2d, "dim") or attention_mask_2d.dim() != 2:
        return None

    bsz, q_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    key_len = int(attention_mask_2d.shape[1])
    dev = input_ids.device
    llm = model.language_model.model
    block = int(getattr(llm, "block_size", N_FUTURE))
    causal_attn = bool(getattr(llm, "causal_attn", False))
    use_mtp_window = bool(mtp_window and q_len >= block and key_len >= block)
    q0 = max(0, q_len - block)
    k0 = max(0, key_len - block)
    blocked_k = k0 - 1
    past_len = int(past_len)

    key_valid = attention_mask_2d.detach().to(device="cpu", dtype=torch.bool).contiguous().numpy()
    key_idx = np.arange(key_len)
    q_ranges, k_ranges, types = [], [], []
    if not use_mtp_window:
        causal_q_ranges, causal_k_ranges, causal_types = [], [], []
        causal_fast_path = True
        packed_flash = True
        for b in range(bsz):
            valid = np.flatnonzero(key_valid[b])
            if valid.size == 0:
                causal_fast_path = False
                break
            valid_len = int(valid[-1]) + 1
            if valid_len < q_len or not bool(key_valid[b, :valid_len].all()) or bool(key_valid[b, valid_len:].any()):
                causal_fast_path = False
                break
            packed_flash = packed_flash and valid_len == key_len
            q_base = b * q_len
            k_base = b * key_len
            causal_q_ranges.append([q_base, q_base + q_len])
            causal_k_ranges.append([k_base, k_base + valid_len])
            causal_types.append(1)
        if causal_fast_path:
            plan = {
                "q_ranges": torch.tensor(causal_q_ranges, dtype=torch.int32, device=dev).contiguous(),
                "k_ranges": torch.tensor(causal_k_ranges, dtype=torch.int32, device=dev).contiguous(),
                "attn_type_map": torch.tensor(causal_types, dtype=torch.int32, device=dev).contiguous(),
                "max_q_len": q_len,
                "max_k_len": max((end - start for start, end in causal_k_ranges), default=0),
                "_la_flash_batched": True,
                "_la_flash_disjoint_q_ranges": True,
            }
            if packed_flash:
                plan.update(
                    {
                        "flash_cu_seqlens_q": torch.arange(
                            0,
                            (bsz + 1) * q_len,
                            q_len,
                            dtype=torch.int32,
                            device=dev,
                        ),
                        "flash_cu_seqlens_k": torch.arange(
                            0,
                            (bsz + 1) * key_len,
                            key_len,
                            dtype=torch.int32,
                            device=dev,
                        ),
                        "flash_causal": True,
                    }
                )
            plan.update(_la_flash_group_plan_tensors(causal_q_ranges, causal_types, dev))
            _record_sparse_plan_stats(model, causal_q_ranges, causal_k_ranges, causal_types)
            return plan

    def row_segments(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return ((0, 1),)
        split = np.flatnonzero(np.diff(idx) > 1) + 1
        starts = np.concatenate((idx[:1], idx[split]))
        ends = np.concatenate((idx[split - 1], idx[-1:])) + 1
        return tuple((int(s), int(e)) for s, e in zip(starts, ends))

    for b in range(bsz):
        q_base = b * q_len
        k_base = b * key_len
        run_start = 0
        run_segments = None
        if use_mtp_window and not causal_attn:
            prefix_q_len = q0
            prefix_k_end = past_len + prefix_q_len
            prefix_ok = (
                prefix_q_len > 0
                and prefix_k_end <= key_len
                and bool(key_valid[b, :prefix_k_end].all())
            )
            window_prefix_ok = blocked_k <= 0 or bool(key_valid[b, :blocked_k].all())
            window_ok = bool(key_valid[b, k0:key_len].all())
            if prefix_ok:
                q_ranges.append([q_base, q_base + prefix_q_len])
                k_ranges.append([k_base, k_base + prefix_k_end])
                types.append(1)
                run_start = prefix_q_len
            if run_start == prefix_q_len and prefix_q_len < q_len and window_prefix_ok and window_ok:
                if blocked_k > 0:
                    q_ranges.append([q_base + prefix_q_len, q_base + q_len])
                    k_ranges.append([k_base, k_base + blocked_k])
                    types.append(0)
                q_ranges.append([q_base + prefix_q_len, q_base + q_len])
                k_ranges.append([k_base + k0, k_base + key_len])
                types.append(0)
                continue

        for q in range(run_start, q_len):
            visible = key_valid[b] & (key_idx <= q + past_len)
            if use_mtp_window and q >= q0:
                if not causal_attn:
                    visible = visible.copy()
                    visible[k0:key_len] = key_valid[b, k0:key_len]
                if blocked_k >= 0:
                    if visible.base is None:
                        visible[blocked_k] = False
                    else:
                        visible = visible.copy()
                        visible[blocked_k] = False
            segments = row_segments(visible)
            if run_segments is None:
                run_segments = segments
                continue
            if segments == run_segments:
                continue
            for start, end in run_segments:
                q_ranges.append([q_base + run_start, q_base + q])
                k_ranges.append([k_base + start, k_base + end])
                types.append(0)
            run_start = q
            run_segments = segments
        for start, end in run_segments:
            q_ranges.append([q_base + run_start, q_base + q_len])
            k_ranges.append([k_base + start, k_base + end])
            types.append(0)

    seen_q_ranges = set()
    disjoint_q_ranges = True
    for start, end in q_ranges:
        key = (int(start), int(end))
        if key in seen_q_ranges:
            disjoint_q_ranges = False
            break
        seen_q_ranges.add(key)

    plan = {
        "q_ranges": torch.tensor(q_ranges, dtype=torch.int32, device=dev).contiguous(),
        "k_ranges": torch.tensor(k_ranges, dtype=torch.int32, device=dev).contiguous(),
        "attn_type_map": torch.tensor(types, dtype=torch.int32, device=dev).contiguous(),
        "max_q_len": max((end - start for start, end in q_ranges), default=0),
        "max_k_len": max((end - start for start, end in k_ranges), default=0),
        "_la_flash_batched": True,
        "_la_flash_disjoint_q_ranges": disjoint_q_ranges,
    }
    plan.update(_la_flash_group_plan_tensors(q_ranges, types, dev))
    _record_sparse_plan_stats(model, q_ranges, k_ranges, types)
    return plan


def _direct_base_forward(
    base,
    input_ids=None,
    visual_features=None,
    image_token_index=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    mod = importlib.import_module(type(base).__module__)
    output_attentions = output_attentions if output_attentions is not None else base.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else base.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else base.config.use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    if input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    past_key_values_length = 0
    use_legacy_cache = False
    if use_cache:
        Cache = getattr(mod, "Cache")
        DynamicCache = getattr(mod, "DynamicCache")
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        dev = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=dev,
        ).unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = base.image_processing(input_ids, visual_features, image_token_index)

    hidden_states = inputs_embeds
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in base.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = base.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    return SimpleNamespace(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def language_model_forward(model, **kwargs):
    """Forward through the text LM, bypassing official dense-mask prep for sparse plans."""
    lm = model.language_model
    return_logits = kwargs.pop("return_logits", True)
    logits_slice = kwargs.pop("logits_slice", None)
    attention_mask = kwargs.get("attention_mask")
    use_direct_sparse = (
        getattr(model, "_la_flash_requested_attn", ATTN_MODE) in {"magi", "la_flash"}
        and _is_magi_plan(attention_mask)
    )
    if not use_direct_sparse:
        return lm(**kwargs)

    labels = kwargs.pop("labels", None)
    if labels is not None:
        raise NotImplementedError("labels are not supported in the direct sparse-plan decode forward")
    output_attentions = kwargs.get("output_attentions", None)
    output_hidden_states = kwargs.get("output_hidden_states", None)
    base_out = _direct_base_forward(lm.model, **kwargs)
    logits = None
    if return_logits:
        hidden_states = base_out.last_hidden_state
        if logits_slice is not None:
            hidden_states = hidden_states[:, logits_slice, :]
        logits = lm.lm_head(hidden_states).float()
    return SimpleNamespace(
        logits=logits,
        past_key_values=base_out.past_key_values,
        hidden_states=base_out.hidden_states if output_hidden_states else None,
        attentions=base_out.attentions if output_attentions else None,
    )


_EagerCls = _SdpaCls = _LaFlashCls = _MagiCls = None
def _attn_classes(mode=None):
    """Attention classes from the dynamic Qwen2 remote module.

    The official Qwen2Model mask dispatcher only implements ``sdpa`` and
    single-row ``magi``.  Eager, LA Flash, and batched Magi inference
    therefore swap the layer class while keeping the model's mask dispatcher
    pinned to ``sdpa``.
    """
    global _EagerCls, _SdpaCls, _LaFlashCls, _MagiCls
    mode = _normalize_attn_mode(mode) if mode is not None else None
    if _SdpaCls is None:
        mod = importlib.import_module(type(_model.language_model.model).__module__)
        _EagerCls = mod.Qwen2Attention
        _SdpaCls = mod.Qwen2SdpaAttention
    else:
        mod = importlib.import_module(type(_model.language_model.model).__module__)
    if (mode is None or mode == "la_flash") and _LaFlashCls is None:
        _LaFlashCls = build_la_flash_attention_class(mod)
    if (mode is None or mode == "magi") and _MagiCls is None:
        _MagiCls = build_batched_magi_attention_class(mod) if getattr(mod, "_MAGI_AVAILABLE", False) else None
    return _EagerCls, _SdpaCls, _LaFlashCls, _MagiCls

def _set_llm_mode(model, mode):
    """Swap every Qwen2 decoder layer's attention class.

    Release backends keep ``Qwen2Model._attn_implementation='sdpa'`` so the
    official Qwen2 mask dispatcher stays available for dense-mask modes. The
    local ``la_flash`` and batched ``magi`` wrappers can also consume scheduler-built
    sparse plans directly, avoiding repeated per-layer dense mask conversion.
    """
    mode = _normalize_attn_mode(mode)
    eager, sdpa, la_flash, magi = _attn_classes(mode)
    impl = "sdpa"
    if mode == "sdpa":
        cls = sdpa
    elif mode == "eager":
        cls = eager
    elif mode == "la_flash":
        cls = la_flash
    elif mode == "magi":
        if magi is None:
            raise RuntimeError("MagiAttention is unavailable in the current Python environment.")
        cls = magi
    else:
        raise ValueError(f"unknown LLM attention mode: {mode}")
    llm = model.language_model.model
    for lyr in llm.layers:
        lyr.self_attn.__class__ = cls
        if mode == "magi":
            lyr.self_attn.softmax_scale = lyr.self_attn.head_dim ** -0.5
    llm._attn_implementation = impl
    llm.config._attn_implementation = llm._attn_implementation
    if hasattr(model.config, "text_config"):
        model.config.text_config._attn_implementation = llm._attn_implementation
    model.config._attn_implementation = llm._attn_implementation
    model._la_flash_requested_attn = mode

_st = _hp = None
def _helpers():
    """The model's own sample_tokens / handle_pattern (the exact box decoders)."""
    global _st, _hp
    if _st is None:
        m = importlib.import_module(type(load()[2]).__module__)
        _st, _hp = m.sample_tokens, m.handle_pattern
    return _st, _hp


_gu = None
def _gen_utils():
    """The model's generate_utils module (apply_repetition_penalty / top_p_logits / top_k_logits /
    decode_bbox_avg / decode_ref / dists) -- the pieces sample_tokens_batched reuses verbatim."""
    global _gu
    if _gu is None:
        m = importlib.import_module(type(load()[2]).__module__)
        _gu = importlib.import_module(m.sample_tokens.__module__)
    return _gu


def _env_float(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return float(default)
    return float(val)


def _coord_fallback_mode():
    mode = os.environ.get("LA_FLASH_COORD_FALLBACK_MODE", "legacy").strip().lower().replace("-", "_")
    aliases = {
        "": "legacy",
        "official": "legacy",
        "range": "legacy",
        "spread": "legacy",
        "none": "off",
        "disable": "off",
        "disabled": "off",
        "entropy_variance": "uncertainty",
        "entropy_var": "uncertainty",
        "ent_var": "uncertainty",
        "entropy_std": "uncertainty",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"legacy", "uncertainty", "off"}:
        raise ValueError(
            "LA_FLASH_COORD_FALLBACK_MODE must be one of legacy, uncertainty, off"
        )
    return mode


def _coord_uncertainty_threshold(coord_start_token_id, coord_end_token_id):
    """Return the coord uncertainty threshold in raw coord-token units.

    Backward-compatible behavior:
    - LA_FLASH_COORD_UNCERTAINTY_THRESH > 1 is treated as raw coord-token RMSE.
    - LA_FLASH_COORD_UNCERTAINTY_THRESH <= 1 is treated as normalized by coord span.
    - LA_FLASH_COORD_UNCERTAINTY_NORM_THRESH is an explicit normalized override.
    """
    coord_span = max(float(coord_end_token_id - coord_start_token_id + 1), 1.0)
    norm_val = os.environ.get("LA_FLASH_COORD_UNCERTAINTY_NORM_THRESH")
    if norm_val is not None and norm_val.strip() != "":
        return float(norm_val) * coord_span

    val = os.environ.get("LA_FLASH_COORD_UNCERTAINTY_THRESH")
    if val is None or val.strip() == "":
        return 20.0
    threshold = float(val)
    if 0.0 < threshold <= 1.0:
        return threshold * coord_span
    return threshold


def _decode_bbox_with_uncertainty(logits, probs, token_ids, keep_k=4, generation_mode="hybrid"):
    """Decode an MTP box with configurable coord uncertainty fallback.

    The default mode is the official LocateAnything rule.  ``uncertainty`` keeps
    the same frame checks and top-k coord selection, but uses one scalar
    criterion per coordinate: the posterior RMSE of committing to the current
    MAP coordinate among valid coord candidates.  This is the Bayes risk under
    squared coordinate error, so probabilities and token distances are folded
    into one threshold in coordinate-token units.
    """
    gu = _gen_utils()
    mode = _coord_fallback_mode()
    if mode == "legacy" or generation_mode != "hybrid":
        return gu.decode_bbox_avg(logits, probs, token_ids, keep_k=keep_k, generation_mode=generation_mode)

    coord_start_token_id = token_ids["coord_start_token_id"]
    coord_end_token_id = token_ids["coord_end_token_id"]
    box_start_token_id = token_ids["box_start_token_id"]
    box_end_token_id = token_ids["box_end_token_id"]
    none_token_id = token_ids["none_token_id"]
    null_token_id = token_ids["null_token_id"]
    device = logits.device

    box_type = gu.is_valid_box_frame(
        probs,
        token_ids,
        start_thresh=_env_float("LA_FLASH_COORD_BOX_START_THRESH", 0.7),
        end_thresh=_env_float("LA_FLASH_COORD_BOX_END_THRESH", 0.2),
        topk=keep_k,
    )
    if box_type == "empty_box":
        return torch.tensor([
            box_start_token_id,
            none_token_id,
            box_end_token_id,
            null_token_id,
            null_token_id,
            null_token_id,
        ], dtype=torch.long, device=device)
    if box_type == "illegal_box":
        return None

    pos_probs, pos_ids = torch.topk(probs[1:5], k=keep_k, dim=-1)
    valid = (pos_ids >= coord_start_token_id) & (pos_ids <= coord_end_token_id)
    has_valid = valid.any(dim=-1)
    if not has_valid.all():
        return None

    first_valid_idx = valid.long().argmax(dim=-1, keepdim=True)
    first_valid_ids = pos_ids.gather(-1, first_valid_idx).squeeze(-1)
    if mode == "off":
        final_coords = first_valid_ids
    else:
        valid_counts = valid.sum(dim=-1)
        valid_probs = torch.where(valid, pos_probs, torch.zeros_like(pos_probs))
        valid_mass = valid_probs.sum(dim=-1).clamp_min(1e-12)
        weights = valid_probs / valid_mass.unsqueeze(-1)
        coord_values = (pos_ids - coord_start_token_id).to(dtype=torch.float32)
        map_coord = (first_valid_ids - coord_start_token_id).to(dtype=torch.float32)
        uncertainty = (weights * (coord_values - map_coord.unsqueeze(-1)).pow(2)).sum(dim=-1).sqrt()
        is_abnormal = (
            (valid_counts > 1)
            & (uncertainty > _coord_uncertainty_threshold(coord_start_token_id, coord_end_token_id))
        )
        final_coords = torch.where(is_abnormal, torch.tensor(0, device=device), first_valid_ids)

    start_t = torch.tensor([box_start_token_id], dtype=final_coords.dtype, device=device)
    end_t = torch.tensor([box_end_token_id], dtype=final_coords.dtype, device=device)
    return torch.cat([start_t, final_coords, end_t])


def _apply_repetition_penalty_lowmem(logits, generated, repetition_penalty):
    """Apply the stock repetition penalty without allocating a [B, S, V] mask."""
    if repetition_penalty == 1.0:
        return logits
    _, _, vocab_size = logits.shape
    for row in range(logits.shape[0]):
        valid_tokens = generated[row].unique()
        valid_tokens = valid_tokens[(valid_tokens >= 0) & (valid_tokens < vocab_size)]
        if valid_tokens.numel() == 0:
            continue
        row_logits = logits[row, :, valid_tokens]
        logits[row, :, valid_tokens] = torch.where(
            row_logits > 0,
            row_logits / repetition_penalty,
            row_logits * repetition_penalty,
        )
    return logits


def _finite_logit_bounds(dtype):
    finfo = torch.finfo(dtype)
    return finfo.min, finfo.max


def _finite_logits(logits):
    if not logits.dtype.is_floating_point:
        logits = logits.float()
    min_val, max_val = _finite_logit_bounds(logits.dtype)
    return torch.nan_to_num(logits, nan=min_val, posinf=max_val, neginf=min_val)


def _finite_logits_(logits):
    if not logits.dtype.is_floating_point:
        return logits.float()
    min_val, max_val = _finite_logit_bounds(logits.dtype)
    return logits.nan_to_num_(nan=min_val, posinf=max_val, neginf=min_val)


def _top_p_logits_slice_(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    remove = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits.masked_fill_(remove, torch.finfo(logits.dtype).min)
    return logits


def _top_p_logits_(logits, top_p):
    """In-place nucleus filtering with bounded sort workspace.

    The MTP sampler uses logits shaped ``[B, 6, V]``.  Top-p is independent for
    each row and each future position, so filtering one position at a time keeps
    the expensive sorted-index workspace at ``[B, V]`` instead of ``[B, 6, V]``.
    """
    if logits.dim() == 3 and logits.shape[1] > 1:
        for pos in range(logits.shape[1]):
            _top_p_logits_slice_(logits[:, pos, :], top_p)
        return logits
    return _top_p_logits_slice_(logits, top_p)


def _top_k_logits_(logits, top_k):
    """In-place top-k filtering mirroring generate_utils.top_k_logits."""
    top_k = min(int(top_k), logits.size(-1))
    threshold = torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(logits < threshold, torch.finfo(logits.dtype).min)
    return logits


def _safe_probs(filtered_logits):
    """Softmax with CUDA-multinomial-safe cleanup and row-wise argmax fallback."""
    filtered_logits = _finite_logits(filtered_logits)
    probs = torch.softmax(filtered_logits, dim=-1, dtype=torch.float32)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    bad = (~torch.isfinite(row_sum)) | (row_sum <= 0)
    if bool(bad.any().item()):
        fallback = torch.zeros_like(probs)
        fallback.scatter_(-1, filtered_logits.argmax(dim=-1, keepdim=True), 1.0)
        probs = torch.where(bad, fallback, probs)
        row_sum = probs.sum(dim=-1, keepdim=True)
    return probs / row_sum.clamp_min(1.0e-20)


def _sample_top_p_sorted_tokens(logits, top_p):
    """Sample from top-p filtered logits without scattering back to vocab order."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits.masked_fill_(remove, torch.finfo(sorted_logits.dtype).min)
    sorted_probs = _safe_probs(sorted_logits)
    sample_idx = sorted_probs.argmax(dim=-1)
    try:
        sample_idx = torch.distributions.Categorical(probs=sorted_probs).sample()
    except Exception:
        pass
    return sorted_indices.gather(-1, sample_idx.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_tokens_batched(logits, generated, token_ids, per_row_temp,
                          repetition_penalty=1.0, top_p=None, top_k=None,
                          keep_k_avg=4, generation_mode='fast'):
    """Batched fork of generate_utils.sample_tokens for the MTP window [B,6,V]. The logits pipeline
    (rep-penalty / per-row temperature / top_p / top_k / softmax / sample) is ROW-INDEPENDENT, so run
    it ONCE over the whole batch instead of B times on [1,6,V] (the per-row san defeats batching by
    slicing wlogits[b:b+1]). Only the variable-length box ASSEMBLY (decode_bbox_avg -> ragged shapes,
    where sample_tokens' final torch.stack throws) stays per-row, returned as a LIST.

    Equivalence to per-row san: every pipeline op reduces on dim=-1 only (never crosses the row dim),
    so row b's processed logits/probs are bit-identical to slicing first -> greedy (per_row_temp==0,
    argmax branch, no RNG) is BIT-EXACT. Under sampling, one batched Categorical changes the global
    RNG consumption order vs B per-row draws -> box-size jitter (blessed; greedy is the exact gate).
    apply_repetition_penalty already loops per-row internally, so passing the full [B,M] `generated`
    is row-correct. keep_k_avg/generation_mode mirror sample_tokens' decode_bbox_avg call EXACTLY
    (note: the per-row san passes keep_k=5 but decode_bbox_avg reads keep_k_avg, default 4 -- so 5 is
    a no-op there; we replicate keep_k_avg=4). Returns (x0[B,6], boxes: list of B 1-D LongTensors)."""
    gu = _gen_utils()
    B, S, V = logits.shape                                        # S = N_FUTURE = 6
    if repetition_penalty != 1.0:
        logits = _apply_repetition_penalty_lowmem(logits, generated, repetition_penalty)
    t = per_row_temp.to(dtype=logits.dtype).view(B, 1, 1)
    sample_rows = per_row_temp > 0
    if bool(sample_rows.all().item()):
        logits.div_(t.clamp(min=1e-8))
    elif bool(sample_rows.any().item()):
        idx = sample_rows.nonzero(as_tuple=True)[0]
        logits[idx].div_(t[idx].clamp(min=1e-8))
    logits = _finite_logits_(logits)
    if top_p is not None and top_p < 1:
        logits = _top_p_logits_(logits, top_p)
    if top_k is not None and top_k > 0:
        logits = _top_k_logits_(logits, top_k)
    probs = _safe_probs(logits)
    x0 = probs.argmax(dim=-1)                                     # [B,6]; greedy rows are final here
    samp = per_row_temp > 0
    if bool(samp.any()):                                         # sampling rows: ONE batched Categorical draw
        idx = samp.nonzero(as_tuple=True)[0]
        try:
            x0[idx] = gu.dists.Categorical(probs=probs[idx]).sample()
        except Exception:
            pass                                                # keep argmax (matches san's except: probs.max)
    boxes = []
    fallback = torch.zeros(1, dtype=x0.dtype, device=x0.device)
    for b in range(B):                                          # variable-length box assembly (per-row, exact)
        db = _decode_bbox_with_uncertainty(
            logits[b], probs[b], token_ids,
            keep_k=keep_k_avg, generation_mode=generation_mode)
        if db is not None:
            boxes.append(db)
        else:
            ref = gu.decode_ref(logits[b], probs[b], token_ids)
            if ref is None:
                boxes.append(fallback)
            elif torch.is_tensor(ref):
                boxes.append(ref.to(dtype=x0.dtype, device=x0.device))
            else:
                boxes.append(torch.tensor(ref, dtype=x0.dtype, device=x0.device))
    return x0, boxes


@torch.no_grad()
def sample_next_tokens_batched(logits, generated, per_row_temp,
                               repetition_penalty=1.0, top_p=None, top_k=None):
    """Batched one-token sampler for AR repair rows.

    This mirrors the row-independent part of ``sample_tokens`` for logits shaped
    ``[B,1,V]``.  It intentionally does not run bbox/ref assembly because AR mode
    only needs the next token before the state machine classifies it.
    """
    gu = _gen_utils()
    if logits.dim() != 3 or logits.shape[1] != 1:
        raise ValueError(f"AR batched sampler expects logits [B,1,V], got {tuple(logits.shape)}")
    B = int(logits.shape[0])
    if repetition_penalty != 1.0:
        logits = _apply_repetition_penalty_lowmem(logits, generated, repetition_penalty)
    t = per_row_temp.to(dtype=logits.dtype).view(B, 1, 1)
    sample_rows = per_row_temp > 0
    if bool(sample_rows.all().item()):
        logits.div_(t.clamp(min=1e-8))
    elif bool(sample_rows.any().item()):
        idx = sample_rows.nonzero(as_tuple=True)[0]
        logits[idx].div_(t[idx].clamp(min=1e-8))
    logits = _finite_logits_(logits)
    sorted_top_p = os.environ.get("AR_SORTED_TOPP", "0") == "1"
    default_top_p = sorted_top_p and top_p is not None and top_p < 1 and (top_k is None or top_k <= 0)
    if default_top_p and bool(sample_rows.all().item()):
        return _sample_top_p_sorted_tokens(logits, top_p)
    if top_p is not None and top_p < 1:
        logits = _top_p_logits_(logits, top_p)
    if top_k is not None and top_k > 0:
        logits = _top_k_logits_(logits, top_k)
    probs = _safe_probs(logits)
    x0 = probs.argmax(dim=-1)
    if bool(sample_rows.any().item()):
        # Keep row-ordered sampling as the release default.  A single batched
        # Categorical is faster, but it consumes RNG differently from stock AR
        # repair and can alter default-temperature termination behavior.
        for row in sample_rows.nonzero(as_tuple=True)[0].tolist():
            try:
                x0[row : row + 1] = gu.dists.Categorical(probs=probs[row : row + 1]).sample()
            except Exception:
                pass
    return x0


def load_pil(p):
    from PIL import Image
    im = Image.open(p).convert("RGB"); w, h = im.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / max(w, h); im = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS)
    return im

def _preproc_one(im):
    """CPU-side processor for one image -> (pixel_values[bf16], grid[int32]). Split out of
    _encode_image so _encode_images can batch the GPU encode while preprocessing stays per-image."""
    tok, proc, model = load()
    msg = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "x"}]}]
    text = proc.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs, vids = proc.process_vision_info(msg)
    inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(DEV)
    grid = inp.get("image_grid_hws")
    if isinstance(grid, np.ndarray): grid = torch.from_numpy(grid).to(DEV, dtype=torch.int32)
    return inp["pixel_values"].to(DT), grid


def _vision_is_flash():
    """True iff MoonViT will actually run flash_attn_varlen (so cross-image packing is
    block-diagonal = exact AND a win). If the vision blocks are on sdpa/eager, OR the flash
    wheel is absent (multihead_attention falls back to the dense-mask sdpa path), packing is
    O(S^2) N^2 -> caller must stay per-image."""
    vm = load()[2].vision_model
    mod = importlib.import_module(type(vm).__module__)
    if getattr(mod, "flash_attn_varlen_func", None) is None:
        return False
    try:
        return vm.encoder.blocks[0].attn_implementation == "flash_attention_2"
    except Exception:
        return False


@torch.no_grad()
def _encode_images(ims):
    """N images -> list of [n_img_tokens, C] mlp1-projected visual_features, one per image
    (row-order). Drop-in for [_encode_image(im) for im in ims].

    With flash present (_vision_is_flash) and N>1, packs images into
    extract_feature micro-batches: MoonViT's varlen cu_seqlens path is
    block-diagonal by image. Without flash, the dense SDPA fallback would scale
    with the packed total sequence length, so this function falls back to
    per-image encode. MTP_BATCH_VISION=0 also forces per-image encode."""
    tok, proc, model = load()
    pvs, grids = [], []
    for im in ims:
        pv, g = _preproc_one(im)
        pvs.append(pv); grids.append(g)
    if BATCH_VISION and len(ims) > 1 and _vision_is_flash():
        if VISION_ENCODE_BATCH_SIZE <= 0 or VISION_ENCODE_BATCH_SIZE >= len(ims):
            vit_list = model.extract_feature(torch.cat(pvs, dim=0), torch.cat(grids, dim=0))
        else:
            vit_list = []
            for start in range(0, len(ims), VISION_ENCODE_BATCH_SIZE):
                end = min(start + VISION_ENCODE_BATCH_SIZE, len(ims))
                vit_list.extend(
                    model.extract_feature(
                        torch.cat(pvs[start:end], dim=0),
                        torch.cat(grids[start:end], dim=0),
                    )
                )
        return [model.mlp1(v) for v in vit_list]        # one [P_i, C] per image (patch_merger split)
    return [model.mlp1(torch.cat(model.extract_feature(pv, g), dim=0))
            for pv, g in zip(pvs, grids)]                # per-image (flash absent / N==1 / forced off)


@torch.no_grad()
def _encode_image(im):
    """Single-image convenience wrapper (single-image callers); = _encode_images([im])[0]
    (takes the per-image path inside _encode_images, so bit-identical to the original)."""
    return _encode_images([im])[0]

@torch.no_grad()
def _tokenize(im, query):
    """1-D prompt token ids for (image, query). Uses the model's own chat template."""
    tok, proc, model = load()
    msg = [{"role": "user", "content": [{"type": "image", "image": im},
                                        {"type": "text", "text": _PROMPT + query + "."}]}]
    text = proc.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs, vids = proc.process_vision_info(msg)
    return proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(DEV)["input_ids"][0]


@torch.no_grad()
def _tokenize_cached_image(query, image_token_count, im=None):
    """Tokenize a prompt when the image token count is already known.

    This keeps the processor's chat template, but directly expands ``<image-1>``
    from the cached visual feature length.  It avoids re-running the CPU image
    processor for every category prompt that shares the same image.
    """
    tok, proc, model = load()
    msg = [{"role": "user", "content": [{"type": "image", "image": im},
                                        {"type": "text", "text": _PROMPT + query + "."}]}]
    text = proc.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    placeholder = f"<{getattr(proc, 'image_placeholder', 'image')}-1>"
    image_token = getattr(proc, "image_token", "<IMG_CONTEXT>")
    image_start = getattr(proc, "image_start_token", "<img>")
    image_end = getattr(proc, "image_end_token", "</img>")
    replacement = f"<image 1>{image_start}{image_token * int(image_token_count)}{image_end}"
    if placeholder not in text:
        raise ValueError(f"cached image placeholder {placeholder!r} was not found in chat template")
    text = text.replace(placeholder, replacement, 1)
    return tok([text], return_tensors="pt").to(DEV)["input_ids"][0]


def _proc_full(im, query):
    """Full processor dict (input_ids, attention_mask, pixel_values, image_grid_hws) —
    used by the bench to drive the STOCK generate for the equivalence check."""
    tok, proc, model = load()
    msg = [{"role": "user", "content": [{"type": "image", "image": im},
                                        {"type": "text", "text": _PROMPT + query + "."}]}]
    text = proc.py_apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    imgs, vids = proc.process_vision_info(msg)
    inp = proc(text=[text], images=imgs, videos=vids, return_tensors="pt").to(DEV)
    grid = inp.get("image_grid_hws")
    if isinstance(grid, np.ndarray): grid = torch.from_numpy(grid).to(DEV, dtype=torch.int32)
    inp["image_grid_hws"] = grid
    return inp

def _pad_generated(prompt_ids, gen_ids, img_tok, dev):
    """Per-row [prompt + accepted] left-padded with the image token (already in every
    prompt -> .unique() unchanged -> repetition penalty identical to single-run)."""
    rows = [list(prompt_ids[b].tolist()) + gen_ids[b] for b in range(len(prompt_ids))]
    M = max(len(r) for r in rows)
    out = torch.full((len(rows), M), img_tok, dtype=torch.long, device=dev)
    for b, r in enumerate(rows):
        out[b, M - len(r):] = torch.tensor(r, dtype=torch.long, device=dev)
    return out
