# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import numpy as np
import random

from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
    get_2d_sincos_pos_embed,
    PixArtAlphaTextProjection,
)
from einops import rearrange

class CombinedTimestepGuidanceEmbedding(nn.Module):
    """
    copy-paste from diffusers.models.embeddings.CombinedTimestepLabelEmbeddings
    with following modifications:
    1. remove `self.class_embedder = LabelEmbedding`
    """
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, guidance, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb + guidance_emb

class CombinedTimestepGuidanceTextProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + guidance_emb + pooled_projections

        return conditioning

class PlainTimestepEmbedding(nn.Module):
    """
    copy-paste from diffusers.models.embeddings.CombinedTimestepLabelEmbeddings
    with following modifications:
    1. remove `self.class_embedder = LabelEmbedding`
    """
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb

class PatchEmbed3D(nn.Module):
    """3D Video to Patch Embedding. But it is a pseudo implementation. i.e, no temporal information is used."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        num_latent_frames=8,
        use_pose_embed=False,
        pos_embed_type="sincos",
        pos_embed_max_size=None,
    ):
        super().__init__()
        self.use_pose_embed = use_pose_embed
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        grid_size = base_size = (num_latent_frames, self.height, self.width)

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        # else:
        #     grid_size = int(num_patches**0.5)

        if use_pose_embed:
            # pos_embed2d = get_2d_sincos_pos_embed(
            #     embed_dim, (self.height, self.width), base_size=self.base_size, interpolation_scale=self.interpolation_scale
            # )
            # # (1 1 S C)
            # self.register_buffer("pos_embed2d", torch.from_numpy(pos_embed2d).float().unsqueeze(0).unsqueeze(1), persistent=False)
            # #  # (1 T 1 C)
            # self.time_pos_embed = nn.Parameter(torch.zeros(1, num_latent_frames, 1, embed_dim).float(), requires_grad=True)

            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent, conditional_input=None):
        is_video = latent.ndim == 5

        if self.pos_embed_max_size is not None:
            length, height, width = latent.shape[-3:]
        else:
            length, height, width = latent.shape[-3], latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        
        if is_video:
            _T = latent.shape[2]
            latent = rearrange(latent, "B C T H W -> (B T) C H W")
            latent = self.proj(latent)
            latent = rearrange(latent, "(B T) C H W -> B C T H W", T=_T)
        else:
            latent = self.proj(latent)

        if conditional_input is not None: ##
            latent = latent + conditional_input
            
        if self.flatten:
            if is_video and self.use_pose_embed:
                latent = rearrange(latent, "B C T H W -> (B T) C H W")
            latent = latent.flatten(2).transpose(1, 2)  # (BT)CHW -> (BT)NC
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate positional embeddings if needed.
        if self.use_pose_embed:
            # if self.height != height or self.width != width:
            #     pos_embed = get_2d_pos_embed_gentron(
            #         embed_dim=self.pos_embed2d.shape[-1],
            #         H_tgt=height * 16,  # FIXME: hard-coded VAE downsampling factor * patch_size
            #         W_tgt=width * 16,
            #     )
            #     pos_embed = torch.from_numpy(pos_embed)
            #     pos_embed = pos_embed.float().unsqueeze(0).unsqueeze(0).to(latent.device)
            # else:
            #     pos_embed = self.pos_embed2d

            # pos_embed = pos_embed.repeat(1, 1, length, 1).to(latent.dtype) #+ self.time_pos_embed
            # pos_embed = rearrange(pos_embed, "B T S C -> B (T S) C")
            # latent = latent + pos_embed

            # Interpolate or crop positional embeddings as needed
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height, width)
            else:
                if self.height != height or self.width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                    )
                    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
                else:
                    pos_embed = self.pos_embed

            latent = latent + pos_embed

            if is_video:
                latent = rearrange(latent, "(B T) N C -> B (T N) C", T=_T)

        return latent.to(latent.dtype), height, width

class RotaryEmbedding1D(nn.Module):
    # https://spaces.ac.cn/archives/8397
    def __init__(
        self,
        dim,
        max_frames=32,# TODO: hardcoded
        base=10000.0,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.max_frames = max_frames
        self.base = base
        self.device = device
        self.scaling_factor = scaling_factor

        assert dim % 2 == 0 # split for sin,cos
        half_dim = dim // 2

        self.n_dims = 1 # split 1 dims for x
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, self.n_dims, dtype=torch.int64).float().to(device) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t_frames = torch.arange(self.max_frames, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # Unified-IO2 style: https://github.com/allenai/unified-io-2.pytorch/blob/733fae901ad0f040b3017d33c8547e0db129a527/uio2/layers.py#L74
        grid3d = torch.stack(torch.meshgrid(t_frames, indexing="ij"), dim=-1).reshape(-1, self.n_dims)
        grid3d = grid3d / self.scaling_factor

        freqs_0 = torch.outer(grid3d[:, 0], self.inv_freq)

        idx_theta = torch.concat([freqs_0], dim=-1)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache = torch.reshape(cache, [max_frames, 1, half_dim, 2])  # t, 128, 128, 1, 36, 2
        self.register_buffer("cache", cache.to(torch.get_default_dtype()), persistent=False)

    def forward(self, x, num_frames, height, width):
        raise NotImplementedError


class RotaryEmbedding2D(nn.Module):
    # https://spaces.ac.cn/archives/8397
    def __init__(
        self,
        dim,
        max_height=32,# TODO: hardcoded
        max_width=32,
        base=10000.0,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.max_height = max_height
        self.max_width = max_width
        self.base = base
        self.device = device
        self.scaling_factor = scaling_factor

        assert dim % 2 == 0 # split for sin,cos
        half_dim = dim // 2

        self.n_dims = 2 # split 2 dims for x,y
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, self.n_dims, dtype=torch.int64).float().to(device) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t_height = torch.arange(self.max_height, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t_width = torch.arange(self.max_width, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # Unified-IO2 style: https://github.com/allenai/unified-io-2.pytorch/blob/733fae901ad0f040b3017d33c8547e0db129a527/uio2/layers.py#L74
        grid3d = torch.stack(torch.meshgrid(t_height, t_width, indexing="ij"), dim=-1).reshape(-1, self.n_dims)
        grid3d = grid3d / self.scaling_factor

        freqs_0 = torch.outer(grid3d[:, 0], self.inv_freq)
        freqs_1 = torch.outer(grid3d[:, 1], self.inv_freq)

        idx_theta = torch.concat([freqs_0, freqs_1], dim=-1)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache = torch.reshape(cache, [max_height, max_width, 1, half_dim, 2])  # t, 128, 128, 1, 36, 2
        self.register_buffer("cache", cache.to(torch.get_default_dtype()), persistent=False)

    def forward(self, x, num_frames, height, width):
        raise NotImplementedError


class RotaryEmbedding3D(nn.Module):
    # https://spaces.ac.cn/archives/8397
    def __init__(
        self,
        dim,
        max_frames=64,
        max_height=64, # TODO: hardcoded
        max_width=64,
        base=10000.0,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.max_frames = max_frames
        self.max_height = max_height
        self.max_width = max_width
        self.base = base
        self.device = device
        self.scaling_factor = scaling_factor

        assert dim % 2 == 0 # split for sin,cos
        half_dim = dim // 2

        self.n_dims = 3 # split 3 dims for x,y,z
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, self.n_dims, dtype=torch.int64).float().to(device) / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t_frames = torch.arange(self.max_frames, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t_height = torch.arange(self.max_height, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t_width = torch.arange(self.max_width, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # Unified-IO2 style: https://github.com/allenai/unified-io-2.pytorch/blob/733fae901ad0f040b3017d33c8547e0db129a527/uio2/layers.py#L74
        grid3d = torch.stack(torch.meshgrid(t_frames, t_height, t_width, indexing="ij"), dim=-1).reshape(-1, self.n_dims)
        grid3d = grid3d / self.scaling_factor

        freqs_0 = torch.outer(grid3d[:, 0], self.inv_freq)
        freqs_1 = torch.outer(grid3d[:, 1], self.inv_freq)
        freqs_2 = torch.outer(grid3d[:, 2], self.inv_freq)

        idx_theta = torch.concat([freqs_0, freqs_1, freqs_2], dim=-1)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        cache = torch.reshape(cache, [max_frames, max_height, max_width, 1, half_dim, 2])  # t, 128, 128, 1, 36, 2
        self.register_buffer("cache", cache.to(torch.get_default_dtype()), persistent=False)

    def forward(self, x, num_frames, height, width):
        raise NotImplementedError

def get_3d_rotary_pos_embed(
    embed_dim, crops_coords, grid_size, theta: float = 10000, use_real: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")
    start, stop = crops_coords
    grid_size_t, grid_size_h, grid_size_w = grid_size
    grid_h = np.linspace(start[1], stop[1], grid_size_h, endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[2], stop[2], grid_size_w, endpoint=False, dtype=np.float32)
    grid_t = np.linspace(start[0], stop[0], grid_size_t, endpoint=False, dtype=np.float32)

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            grid_size_t, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            grid_size_t, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            grid_size_t * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w
    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis.unbind(-1)  # [S, D] NOTE: Goku stack on the last dimension
        # cos = cos[None, None]
        # sin = sin[None, None]
        if cos.ndim==2 and sin.ndim==2:
            cos = cos[None, None] # [1 1 S D]
            sin = sin[None, None] # [1 1 S D]
        else:
            cos = cos.unsqueeze(1) # [B 1 S D]
            sin = sin.unsqueeze(1) # [B 1 S D]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def get_resize_crop_region_for_grid(src, tgt):
    th, tw = tgt
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def get_resize_crop_region_for_grid_3d(src, tgt):    # src 来源的分辨率   tgt base 分辨率
    tt, th, tw = tgt
    t, h, w = src

    rt, rh, rw = tt/t, th/h, tw/w

    # resize
    resize_length = int(round(min(rt, rh, rw) * t))
    resize_height = int(round(min(rt, rh, rw) * h))
    resize_width = int(round(min(rt, rh, rw) * w))

    crop_start = int(round((tt - resize_length) / 2.0))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    #print(src, tgt, resize_length, resize_height, resize_width, crop_start, crop_top, crop_left)

    return (crop_start, crop_top, crop_left), (crop_start + resize_length, crop_top + resize_height, crop_left + resize_width)

def prepare_rotary_positional_embeddings(
    grid_h: int,
    grid_w: int,
    grid_t: int,
    attention_head_dim: int,
    device: torch.device,
    use_relative: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:    
    if use_relative:
        base_size = (64, 45, 80) # [B C 256 720 1280]
        grid_crops_coords = get_resize_crop_region_for_grid_3d((grid_t, grid_h, grid_w), base_size)
    else:
        grid_crops_coords = ((0, 0, 0), (grid_t, grid_h, grid_w))

    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_t, grid_h, grid_w),
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

if __name__ == "__main__":

    image_rotary_emb = prepare_rotary_positional_embeddings(
        height=480,
        width=864,
        num_frames=13,
        attention_head_dim=64,
        device=torch.device("cuda"),
    )
