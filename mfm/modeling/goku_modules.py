# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

import hashlib
import importlib
import inspect
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import maybe_allow_in_graph
from einops import rearrange

# from torch._inductor.utils import cache_dir
from torch.utils.checkpoint import checkpoint

from mfm.modeling.embeddings import apply_rotary_emb
from mfm.modeling.normalization import AdaLayerNormZeroNoLabel, RMSNorm
from mfm.opendit.utils.comm import (
    AllGather,
    all_to_all_comm,
    gather_sequence,
    split_sequence,
)
from mfm.utils.profiler import CustomFlops, Flops
from mfm.utils.profiler import context_fn as _prof_context_fn

if torch.__version__ >= "2.5.0":
    from torch.utils.checkpoint import (
        create_selective_checkpoint_contexts as _pt2_selective_checkpoint_context_fn_gen,
    )
else:
    from torch.utils.checkpoint import _pt2_selective_checkpoint_context_fn_gen


fa2_installed = importlib.util.find_spec("flash_attn") is not None
fa3_hopper_installed = importlib.util.find_spec("flash_attn_interface") is not None
pt_24_installed = torch.__version__ >= "2.4.0"

if fa3_hopper_installed:
    from flash_attn_interface import flash_attn_varlen_func

    flash_attn_installed = True
elif fa2_installed:
    from flash_attn import flash_attn_varlen_func

    flash_attn_installed = True
else:
    flash_attn_installed = False


def cached_compile(
    model: Callable, enable_cached_compile, compile_mem_gc_threshold, use_navit, *args, **kwargs
):
    if not enable_cached_compile:
        return model(*args, **kwargs)

    if (
        compile_mem_gc_threshold >= 0
        and torch.cuda.memory_allocated() / 2**30 > compile_mem_gc_threshold
    ):
        import gc

        torch.cuda.empty_cache()
        gc.collect()

    if not use_navit:
        inp_args = [v for v in args].extend([v for v in list(kwargs.values())])
        if inp_args is None:
            sha = [None]
        else:
            sha = [(v.shape, v.stride()) if isinstance(v, torch.Tensor) else v for v in inp_args]
            sha = [None if isinstance(v, Callable) else v for v in sha]
        sha.append(inspect.getsourcelines(model)[1])
        md = hashlib.sha224(json.dumps(sha).encode("utf-8")).hexdigest()
        my_dir = os.getenv("DIT_CACHE_DIR", "/tmp/torchinductor_tiger/") + md
        if os.getenv("TORCHINDUCTOR_CACHE_DIR") != my_dir:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = my_dir
            # cache_dir.cache_clear()

    if pt_24_installed:
        from torch._inductor.async_compile import shutdown_compile_workers

        shutdown_compile_workers()
        torch._dynamo.config.inline_inbuilt_nn_modules = True

    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    torch._dynamo.config.automatic_dynamic_shapes = use_navit
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.guard_nn_modules = False
    if torch.__version__ < "2.5.0":
        torch._dynamo.config._experimental_support_context_fn_in_torch_utils_checkpoint = True
    torch._inductor.config.compile_threads = 1

    fn = torch.compile(model, options={"fx_graph_cache": True})
    return fn(*args, **kwargs)


def get_custom_policy():
    no_recompute_list = [
        torch.ops.aten.mm.default,
    ]

    if fa3_hopper_installed:
        no_recompute_list.extend(
            [
                torch.ops.flash_attn._hopper_flash_attn_forward.default,
                torch.ops.flash_attn._hopper_flash_attn_varlen_forward.default,
            ]
        )

    if fa2_installed:
        no_recompute_list.extend(
            [
                torch.ops.flash_attn._flash_attn_forward.default,
            ]
        )

    def custom_policy(mode, func, *args, **kwargs):
        assert mode in ["forward", "recompute"]
        Flops.enable = True if mode == "forward" else False
        return func in no_recompute_list

    return custom_policy


def selective_checkpointing_context_fn():
    # return
    return _pt2_selective_checkpoint_context_fn_gen(get_custom_policy())


def checkpoint_func(mod, *args, **kwargs):
    return checkpoint(
        mod._origin_forward,
        *args,
        **kwargs,
        use_reentrant=False,
        context_fn=mod._context_fn,
    )


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=0.1, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = (
                x.float().mul_(self.gamma.float())
                if self.inplace
                else x.float() * self.gamma.float()
            )
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class PlainMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.fc1 = nn.Linear(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            self.approximate = "tanh"
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        self.fc2 = nn.Linear(inner_dim, dim_out, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(
            dtype=gate.dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Attention(nn.Module, CustomFlops):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor=None,
        out_dim: int = None,
        cross_attention_rope: bool = False,
        qk_normalization: bool = True,
        sequence_parallel_size: int = 0,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.head_dim = dim_head
        self.num_heads = heads

        if (dim_head * heads) != self.inner_dim:
            raise ValueError(
                f"dim_head ({dim_head}) * heads ({heads}) must be equal to inner_dim ({self.inner_dim})."
            )

        # TODO: check bias
        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.o_proj = nn.Linear(self.inner_dim, self.out_dim, bias=bias)

        self.qk_normalization = qk_normalization
        if self.qk_normalization:
            self.q_norm = RMSNorm(self.inner_dim)
            self.k_norm = RMSNorm(self.inner_dim)

        self.sequence_parallel_size = sequence_parallel_size
        if self.sequence_parallel_size > 1:
            self.sequence_parallel_rank = None
            self.sequence_parallel_group = None
            self.sequence_parallel_param_slice = None
            self.local_num_heads = self.num_heads // self.sequence_parallel_size

    def flops(self, args, kwargs, output) -> dict:
        inputs_q = kwargs["inputs_q"]
        inputs_kv = kwargs["inputs_kv"] if kwargs["inputs_kv"] is not None else inputs_q

        if len(inputs_q.shape) == 2:
            # navit mode
            q_bsz = kv_bsz = 1
            q_len = inputs_q.shape[0]
            kv_len = inputs_kv.shape[0]

            cu_seqlens_q = kwargs["cu_seqlens_q"].to(torch.int64).cpu().numpy()
            cu_seqlens_k = kwargs["cu_seqlens_k"].to(torch.int64).cpu().numpy()

            attn_seq_coef = 0
            for i in range(len(cu_seqlens_q) - 1):
                seqlen_q = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                seqlen_k = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                attn_seq_coef += seqlen_q * seqlen_k
        else:
            q_bsz = inputs_q.shape[0]
            q_len = inputs_q.shape[1]
            kv_bsz = inputs_kv.shape[0]
            kv_len = inputs_kv.shape[1]
            attn_seq_coef = q_len * kv_len

        sp_size = self.sequence_parallel_size or 1
        q_proj_flops = q_bsz * q_len * self.inner_dim * self.query_dim * 2
        kv_proj_flops = kv_bsz * kv_len * self.inner_dim * self.cross_attention_dim * 2 * 2
        q_norm_flops = q_bsz * q_len * (2 * self.inner_dim + 1) * 2
        k_norm_flops = kv_bsz * kv_len * (2 * self.inner_dim + 1) * 2
        flash_attention_flops = (
            q_bsz * self.num_heads * attn_seq_coef * self.head_dim * 2 * 2 // sp_size
        )
        o_proj_flops = q_bsz * q_len * self.inner_dim * self.out_dim * 2

        return dict(
            {
                "RMSNorm": q_norm_flops + k_norm_flops,
                "Linear": q_proj_flops + kv_proj_flops + o_proj_flops,
                "Attention": flash_attention_flops,
            }
        )

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.num_heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(
                    padding_shape, dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = torch.nn.functional.pad(
                    attention_mask, (0, target_length), value=0.0
                )

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def post_setup_sequence_parallel(
        self, sequence_parallel_group, sequence_parallel_strategy="long-seq"
    ):
        if self.sequence_parallel_size < 2:
            return
        assert (
            sequence_parallel_group is not None
        ), "sequence_parallel_group must setup before forward"

        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_rank = dist.get_rank(sequence_parallel_group)
        if self.sequence_parallel_param_slice is None:
            self.sequence_parallel_param_slice = slice(
                self.k_proj.out_features
                // self.sequence_parallel_size
                * self.sequence_parallel_rank,
                self.k_proj.out_features
                // self.sequence_parallel_size
                * (self.sequence_parallel_rank + 1),
            )

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        q_height: int,
        q_width: int,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention: bool = False,
        rope_pos_embed=None,
        sequence_parallel_rank=None,  # placeholder, not used
        sequence_parallel_size=1,
        sequence_parallel_group=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        lazy_repeat_times=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.sequence_parallel_size > 1:
            inputs_q = gather_sequence(
                inputs_q,
                process_group=self.sequence_parallel_group,
                dim=0 if max_seqlen_q is not None else 1,
                grad_scale="up",
            )

        if inputs_kv is None:
            inputs_kv = inputs_q

        query_states = self.q_proj(inputs_q)
        key_states = self.k_proj(inputs_kv)
        value_states = self.v_proj(inputs_kv)

        if self.qk_normalization:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if lazy_repeat_times is not None:
            key_states = (
                key_states.unsqueeze(1).expand(-1, lazy_repeat_times, -1, -1).flatten(0, 1)
            )
            value_states = (
                value_states.unsqueeze(1).expand(-1, lazy_repeat_times, -1, -1).flatten(0, 1)
            )
            if attention_mask is not None:
                attention_mask = (
                    attention_mask.unsqueeze(1).expand(-1, lazy_repeat_times, -1, -1).flatten(0, 1)
                )

        if max_seqlen_q is not None:
            # navit mode
            query_states = query_states.view(-1, self.num_heads, self.head_dim)
            key_states = key_states.view(-1, self.num_heads, self.head_dim)
            value_states = value_states.view(-1, self.num_heads, self.head_dim)

            if rope_pos_embed is not None:
                query_states = apply_rotary_emb(
                    query_states.permute(1, 0, 2)[None], rope_pos_embed
                )[0].permute(1, 0, 2)
                if not cross_attention:
                    key_states = apply_rotary_emb(
                        key_states.permute(1, 0, 2)[None], rope_pos_embed
                    )[0].permute(1, 0, 2)

            if fa3_hopper_installed:
                attn_output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=torch.tensor(max_seqlen_q),
                    max_seqlen_k=torch.tensor(max_seqlen_k),
                )
                attn_output = attn_output if pt_24_installed else attn_output[0]
            else:
                attn_output = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                )

            attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            if self.sequence_parallel_size > 1:
                attn_output = split_sequence(attn_output, self.sequence_parallel_group, 0, "down")
            return attn_output

        bsz, q_len, emb_dim = inputs_q.shape
        # assert q_len == q_height * q_width, f"q_len: {q_len}, q_height: {q_height}, q_width: {q_width}"
        kv_len = inputs_kv.shape[1]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        if rope_pos_embed is not None:
            query_states = apply_rotary_emb(query_states, rope_pos_embed)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states, rope_pos_embed)

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, kv_len, bsz)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(bsz, self.num_heads, -1, attention_mask.shape[-1])

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        dtype_before_upcast = query_states.dtype
        if self.upcast_attention:
            if attention_mask is not None:
                assert attention_mask.dtype in [
                    torch.float,
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ], attention_mask.dtype
                attention_mask = attention_mask.to(torch.float32)
            query_states = query_states.to(torch.float32)
            key_states = key_states.to(torch.float32)
            value_states = value_states.to(torch.float32)

        with torch.amp.autocast(
            device_type="cuda", enabled=not self.upcast_attention, dtype=dtype_before_upcast
        ):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.inner_dim)
        attn_output = attn_output.to(dtype_before_upcast)
        attn_output = self.o_proj(attn_output)
        if self.sequence_parallel_size > 1:
            attn_output = split_sequence(attn_output, self.sequence_parallel_group, 1, "down")

        return attn_output


class DistAttention(Attention):
    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        q_height: int,
        q_width: int,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention: bool = False,
        rope_pos_embed=None,
        sequence_parallel_rank=None,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        lazy_repeat_times=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # prepare sequence parallel

        inputs_q = AllGather.apply(inputs_q.contiguous(), self.sequence_parallel_group)[0]
        inputs_q = rearrange(inputs_q, "sp b n c -> b (sp n) c")
        # real here: (B, L, C)
        if inputs_kv is None:
            inputs_kv = inputs_q
        kv_len = inputs_kv.shape[1]

        bsz, q_len, emb_dim = inputs_q.shape
        # print(inputs_q.shape, q_len, self.sequence_parallel_size, q_height, q_width)
        # assert q_len//self.sequence_parallel_size == q_height * q_width, f"q_len: {q_len}, q_height: {q_height}, q_width: {q_width}"

        query_states = F.linear(
            inputs_q,
            self.q_proj.weight[self.sequence_parallel_param_slice],
            self.q_proj.bias[self.sequence_parallel_param_slice],
        )
        key_states = F.linear(
            inputs_kv,
            self.k_proj.weight[self.sequence_parallel_param_slice],
            self.k_proj.bias[self.sequence_parallel_param_slice],
        )
        value_states = F.linear(
            inputs_kv,
            self.v_proj.weight[self.sequence_parallel_param_slice],
            self.v_proj.bias[self.sequence_parallel_param_slice],
        )

        if self.qk_normalization:
            query_states = self.q_norm(query_states, sp_slice=self.sequence_parallel_param_slice)
            key_states = self.k_norm(key_states, sp_slice=self.sequence_parallel_param_slice)

        query_states = query_states.view(
            bsz, q_len, self.local_num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.local_num_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, kv_len, self.local_num_heads, self.head_dim
        ).transpose(1, 2)

        if rope_pos_embed is not None:
            query_states = apply_rotary_emb(query_states, rope_pos_embed)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states, rope_pos_embed)

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, q_len, bsz)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                bsz, self.local_num_heads, -1, attention_mask.shape[-1]
            )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        dtype_before_upcast = query_states.dtype
        if self.upcast_attention:
            if attention_mask is not None:
                assert attention_mask.dtype in [
                    torch.float,
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ], attention_mask.dtype
                attention_mask = attention_mask.to(torch.float32)
            query_states = query_states.to(torch.float32)
            key_states = key_states.to(torch.float32)
            value_states = value_states.to(torch.float32)

        with torch.cuda.amp.autocast(enabled=not self.upcast_attention, dtype=dtype_before_upcast):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.inner_dim // self.sequence_parallel_size
        )
        attn_output = all_to_all_comm(
            attn_output,
            self.sequence_parallel_group,
            scatter_dim=1,
            gather_dim=2,
        )  # (B, L, C/sp) --> (B, L/sp, C)
        attn_output = attn_output.to(dtype_before_upcast)
        attn_output = self.o_proj(attn_output)

        return attn_output


class DistNaviTAttention(Attention):
    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        q_height: int,
        q_width: int,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention: bool = False,
        rope_pos_embed=None,
        sequence_parallel_rank=None,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        lazy_repeat_times=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if inputs_kv is None:
            inputs_kv = inputs_q

        kv_len = inputs_kv.shape[0] * self.sequence_parallel_size
        q_len, emb_dim = inputs_q.shape
        q_len *= self.sequence_parallel_size

        query_states = self.q_proj(inputs_q)
        key_states = self.k_proj(inputs_kv)
        value_states = self.v_proj(inputs_kv)

        # ulysses method
        if self.qk_normalization:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = all_to_all_comm(
            query_states,
            self.sequence_parallel_group,
            1,  # scatter dim
            0,  # gather dim
        )  # (L/sp, C) --> (L, C/sp)
        key_states = all_to_all_comm(
            key_states,
            self.sequence_parallel_group,
            1,  # scatter dim
            0,  # gather dim
        )  # (L/sp, C) --> (L, C/sp)
        value_states = all_to_all_comm(
            value_states,
            self.sequence_parallel_group,
            1,  # scatter dim
            0,  # gather dim
        )  # (L/sp, C) --> (L, C/sp)

        query_states = query_states.view(q_len, self.local_num_heads, self.head_dim).contiguous()
        key_states = key_states.view(kv_len, self.local_num_heads, self.head_dim).contiguous()
        value_states = value_states.view(kv_len, self.local_num_heads, self.head_dim).contiguous()

        if lazy_repeat_times is not None:
            raise NotImplementedError("lazy_repeat_times is not supported in DistNaviTAttention")

        query_states = query_states.view(-1, self.local_num_heads, self.head_dim)
        key_states = key_states.view(-1, self.local_num_heads, self.head_dim)
        value_states = value_states.view(-1, self.local_num_heads, self.head_dim)

        if rope_pos_embed is not None:
            query_states = apply_rotary_emb(query_states.permute(1, 0, 2)[None], rope_pos_embed)[
                0
            ].permute(1, 0, 2)
            if not cross_attention:
                key_states = apply_rotary_emb(key_states.permute(1, 0, 2)[None], rope_pos_embed)[
                    0
                ].permute(1, 0, 2)

        if fa3_hopper_installed:
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=torch.tensor(max_seqlen_q),
                max_seqlen_k=torch.tensor(max_seqlen_k),
            )  # -1, local_num_heads, head_dim
            attn_output = attn_output if pt_24_installed else attn_output[0]
        else:
            attn_output = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )  # -1, local_num_heads, head_dim

        attn_output = all_to_all_comm(
            attn_output,
            self.sequence_parallel_group,
            0,  # scatter dim
            1,  # gather dim
        )  # (L, head/sp, C) --> (L/sp, head, C)
        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


@maybe_allow_in_graph
class TransformerBlockGoku(nn.Module):
    """
    Transformer block.
    Some arguments are not used but are kept for compatibility with the BasicTransformerBlock in diffusers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_zero_nolabel', 'ada_norm_single', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        cross_attention_rope: bool = False,
        adaln_norm_type: str = "rmsnorm",
        sequence_parallel_size=0,  # 0: without sequential parallel. >1: sequential parallel size; 1: FIXME: bug exists for sequence_parallel_size=1
        use_navit: bool = False,
        temporal_modeling=None,
        layer_scale_init_value=None,
        temporal_block_height=1,
        temporal_block_width=1,
        enable_cached_compile: bool = False,
        compile_mem_gc_threshold: int = -1,
        checkpointing: Optional[str] = None,
    ):
        super().__init__()
        self.use_navit = use_navit
        self.checkpointing = checkpointing
        self.temporal_modeling = temporal_modeling
        self.temporal_block_height = temporal_block_height
        self.temporal_block_width = temporal_block_width
        self.enable_cached_compile = enable_cached_compile
        self.compile_mem_gc_threshold = compile_mem_gc_threshold
        if use_navit and not flash_attn_installed:
            raise ImportError("Navit Attention requires `flash-attn`")

        self.context_fn = _prof_context_fn
        self.norm_type = norm_type

        if norm_type in ["ada_norm_zero", "ada_norm_zero_nolabel"]:
            self.norm1 = AdaLayerNormZeroNoLabel(
                dim,
                num_embeds_ada_norm,
                adaln_norm_type,
                use_navit=use_navit,
            )
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
        self.sequence_parallel_size = sequence_parallel_size
        assert (
            sequence_parallel_size != 1
        ), f"FIXME: sequence_parallel_size {sequence_parallel_size} does not work now"
        if sequence_parallel_size > 1:
            if use_navit:
                attn_cls = DistNaviTAttention
            else:
                attn_cls = DistAttention
        else:
            attn_cls = Attention
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,  # WARNING: must set `None` here, which will be parsed as self-attention
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            sequence_parallel_size=sequence_parallel_size,
        )

        self.layer_scale1 = (
            nn.Identity()
            if layer_scale_init_value is None
            else LayerScale(
                num_attention_heads * attention_head_dim,
                init_values=layer_scale_init_value,
                force_fp32=True,
            )
        )

        if cross_attention_dim is not None or double_self_attention:
            # 2. Cross Attention
            self.norm2 = RMSNorm(dim, eps=1e-6)  # TODO: check
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                cross_attention_rope=cross_attention_rope,
                sequence_parallel_size=sequence_parallel_size,
            )
            self.layer_scale2 = (
                nn.Identity()
                if layer_scale_init_value is None
                else LayerScale(
                    num_attention_heads * attention_head_dim,
                    init_values=layer_scale_init_value,
                    force_fp32=True,
                )
            )
        else:
            self.norm2, self.attn2, self.layer_scale2 = None, None, None

        self.norm3 = RMSNorm(dim, eps=1e-6)  # TODO: check
        self.ff = PlainMLP(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self.layer_scale3 = (
            nn.Identity()
            if layer_scale_init_value is None
            else LayerScale(
                num_attention_heads * attention_head_dim,
                init_values=layer_scale_init_value,
                force_fp32=True,
            )
        )

    def post_setup_sequence_parallel(
        self, sequence_parallel_group, sequence_parallel_strategy="long-seq"
    ):
        self.sequence_parallel_group = sequence_parallel_group
        for _, m in self.named_modules():
            if isinstance(m, Attention):
                m.post_setup_sequence_parallel(sequence_parallel_group, sequence_parallel_strategy)

    def forward(self, *args, **kwargs):
        if self.enable_cached_compile and self.use_navit:
            encoder_hidden_states = args[4] if len(args) > 4 else kwargs["encoder_hidden_states"]
            torch._dynamo.mark_dynamic(encoder_hidden_states, 0)

        return cached_compile(
            self.checkpoint_func,
            self.enable_cached_compile,
            self.compile_mem_gc_threshold,
            self.use_navit,
            *args,
            **kwargs,
        )

    def checkpoint_func(self, *args, **kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs["hidden_states"]
        checkpointing = args[11] if len(args) > 11 else kwargs["checkpointing"]

        if (
            self.training
            and checkpointing in ["full-block", "selective"]
            and hidden_states.requires_grad
        ):
            context_fn = (
                selective_checkpointing_context_fn
                if checkpointing == "selective"
                else _prof_context_fn
            )
            return checkpoint(
                self.forward_func, *args, **kwargs, use_reentrant=False, context_fn=context_fn
            )
        else:
            return self.forward_func(*args, **kwargs)

    def forward_func(
        self,
        hidden_states: torch.FloatTensor,
        q_height: int,
        q_width: int,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        rope_cache=None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        checkpointing: Optional[str] = None,
        sequence_parallel_rank=None,
        sequence_parallel_group=None,
        sequence_parallel_strategy="long-seq",
        cumsum_q_len=None,
        cumsum_kv_len=None,
        batch_q_len=None,
        batch_kv_len=None,
        rope_pos_embed=None,
        time_pos_embed=None,
        use_reentrant_checkpoint=False,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states,
            timestep,
            class_labels,
            hidden_dtype=hidden_states.dtype,
            batch_seq_len=batch_q_len,
            sequence_parallel_size=self.sequence_parallel_size,
            sequence_parallel_group=self.sequence_parallel_group,
        )

        # TODO: maybe additional RMSNorm here
        attn_output = self.attn1(
            inputs_q=norm_hidden_states,
            inputs_kv=None,
            q_height=q_height,
            q_width=q_width,
            attention_mask=attention_mask,
            cross_attention=False,
            rope_pos_embed=rope_pos_embed,
            sequence_parallel_rank=sequence_parallel_rank,
            sequence_parallel_size=self.sequence_parallel_size,
            sequence_parallel_group=None,
            cu_seqlens_q=cumsum_q_len,
            cu_seqlens_k=cumsum_q_len,
            max_seqlen_q=max(batch_q_len) if batch_q_len is not None else None,
            max_seqlen_k=max(batch_q_len) if batch_q_len is not None else None,
            lazy_repeat_times=None,
        )

        if self.use_navit:
            attn_output = (gate_msa * attn_output.float()).to(attn_output.dtype)
        else:
            attn_output = (gate_msa.unsqueeze(1) * attn_output.float()).to(attn_output.dtype)

        hidden_states = self.layer_scale1(attn_output) + hidden_states

        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                inputs_q=norm_hidden_states,
                inputs_kv=encoder_hidden_states,
                q_height=q_height,
                q_width=q_width,
                attention_mask=encoder_attention_mask,
                cross_attention=True,
                sequence_parallel_rank=sequence_parallel_rank,
                sequence_parallel_size=self.sequence_parallel_size,
                sequence_parallel_group=sequence_parallel_group,
                rope_pos_embed=rope_pos_embed,
                cu_seqlens_q=cumsum_q_len,
                cu_seqlens_k=cumsum_kv_len,
                max_seqlen_q=max(batch_q_len) if batch_q_len is not None else None,
                max_seqlen_k=max(batch_kv_len) if batch_kv_len is not None else None,
                lazy_repeat_times=None,
            )
            hidden_states = hidden_states + self.layer_scale2(attn_output)

        # Fully Connected
        # TODO: maybe additional RMSNorm here
        norm_hidden_states = self.norm3(hidden_states)
        if self.use_navit:
            norm_hidden_states = (norm_hidden_states.float() * (1 + scale_mlp) + shift_mlp).to(
                norm_hidden_states.dtype
            )
            ff_output = self.ff(norm_hidden_states)
            ff_output = (gate_mlp * ff_output.float()).to(ff_output.dtype)
        else:
            norm_hidden_states = (
                norm_hidden_states.float() * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ).to(norm_hidden_states.dtype)
            ff_output = self.ff(norm_hidden_states)
            ff_output = (gate_mlp.unsqueeze(1) * ff_output.float()).to(ff_output.dtype)
        hidden_states = self.layer_scale3(ff_output) + hidden_states

        return hidden_states
