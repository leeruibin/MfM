# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from mfm.opendit.utils.comm import (
    gather_sequence,
)
from mfm.utils.profiler import CustomFlops


class AdaLayerNormZeroNoLabel(nn.Module):
    """
    copy-paste from diffusers.models.normalization.AdaLayerNormZero
    with following modifications:
    1. remove label embedding (use a diffusion `self.emb`)
    TODO: add additional conditions such as image size and aspect ratio

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim,
        num_embeddings,
        norm_type="layernorm",
        mlp=True,
        use_navit=False,
    ):
        super().__init__()
        self.mlp = mlp
        self.use_navit = use_navit
        self.emb = None

        self.silu = nn.SiLU()
        if self.mlp:
            self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        else:
            self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        if norm_type == "layernorm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "rmsnorm":
            self.norm = RMSNorm(embedding_dim, eps=1e-6)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: torch.LongTensor,
        hidden_dtype: Optional[torch.dtype] = None,
        batch_seq_len: Optional[torch.Tensor] = None,
        sequence_parallel_size=None,
        sequence_parallel_group=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        if self.emb is None:
            emb = self.linear(self.silu(timestep))
        else:
            emb = self.linear(
                self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
            )
        if self.mlp:
            if self.use_navit:
                emb = torch.cat(
                    [embi[None].expand(ri, -1) for embi, ri in zip(emb, batch_seq_len)]
                )
                if sequence_parallel_size is not None and sequence_parallel_size > 1:
                    emb = emb.chunk(sequence_parallel_size, dim=0)[
                        dist.get_rank(sequence_parallel_group)
                    ]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.float().chunk(
                6, dim=1
            )
            if self.use_navit:
                x = self.norm(x).float() * (1 + scale_msa) + shift_msa
            else:
                x = self.norm(x).float() * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x.to(input_dtype), gate_msa, shift_mlp, scale_mlp, gate_mlp
        else:
            if self.use_navit:
                raise NotImplementedError
            shift_msa, scale_msa, gate_msa = emb.float().chunk(3, dim=1)
            x = self.norm(x).float() * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x.to(input_dtype), gate_msa


# https://github.com/huggingface/transformers/blob/2f12e408225b1ebceb0d2f701ce419d46678dc31/src/transformers/models/llama/modeling_llama.py#L76
class RMSNorm(nn.Module, CustomFlops):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.sequence_parallel_group = None

    def forward(self, hidden_states, sp_slice=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)

        if self.sequence_parallel_group is not None:
            variance = gather_sequence(
                variance, process_group=self.sequence_parallel_group, dim=2, grad_scale="up"
            )
            variance = variance.mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if sp_slice is None:
            return (self.weight * hidden_states).to(input_dtype)
        else:
            return (self.weight[sp_slice] * hidden_states).to(input_dtype)

    def flops(self, args, kwargs, output) -> dict:
        hidden_states = args[0]
        if len(hidden_states.shape) == 2:
            # navit mode
            bsz = 1
            seq_len = hidden_states.shape[0]
        else:
            bsz = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]

        return dict(
            {
                "RMSNorm": bsz * seq_len * (2 * self.hidden_size + 1) * 2,
            }
        )
