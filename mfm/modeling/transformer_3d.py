# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

# copy-paste from diffusers/models/transformers/transformer_2d.py
#
import json
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, Optional

import diffusers
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from diffusers.utils import BaseOutput
from torch import nn
from torch.distributed import ProcessGroup

from mfm.modeling.light_adapter import ConvitConditioningEmbedding
from mfm.modeling.embeddings import (
    CombinedTimestepGuidanceEmbedding,
    PatchEmbed3D,
    PlainTimestepEmbedding,
    prepare_rotary_positional_embeddings,
)
from mfm.modeling.goku_modules import TransformerBlockGoku
from mfm.opendit.utils.comm import gather_sequence
from mfm.opendit.utils.operation import gather_forward_split_backward

TF_BLOCKS = {
    "goku": TransformerBlockGoku,
}


@dataclass
class Transformer3DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.block_index_list = list(range(index, index + num_blocks_in_a_chunk))
        self.module = nn.ModuleList()
        for i in range(index, index + num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(
        self,
        hidden_states,
        q_height,
        q_width,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep,
        cross_attention_kwargs,
        class_labels,
        rope_cache,
        added_cond_kwargs,
        sequence_parallel_rank,
        sequence_parallel_group,
        sequence_parallel_strategy,
        cumsum_q_len,
        cumsum_kv_len,
        batch_q_len,
        batch_kv_len,
        rope_pos_embed,
        time_pos_embed,
        use_reentrant_checkpoint=False,
        checkpointing="full-block",  # None
    ):
        for blk_idx, m in zip(self.block_index_list, self.module):
            hidden_states = m(
                hidden_states,
                q_height=q_height,
                q_width=q_width,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states[blk_idx % 2],
                encoder_attention_mask=encoder_attention_mask[blk_idx % 2],
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                rope_cache=rope_cache,
                added_cond_kwargs=added_cond_kwargs,
                checkpointing=checkpointing,
                sequence_parallel_rank=sequence_parallel_rank,
                sequence_parallel_group=None,
                sequence_parallel_strategy=sequence_parallel_strategy,
                cumsum_q_len=cumsum_q_len,
                cumsum_kv_len=cumsum_kv_len,
                batch_q_len=batch_q_len,
                batch_kv_len=batch_kv_len,
                rope_pos_embed=rope_pos_embed,
                time_pos_embed=time_pos_embed,
                use_reentrant_checkpoint=use_reentrant_checkpoint,
            )
        return hidden_states


class Transformer3DModel(ModelMixin, ConfigMixin):
    """
    A 3D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    _supports_gradient_checkpointing = None

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        cond_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        patch_size: Optional[int] = 2,
        patch_size_t: Optional[int] = 1,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        num_latent_frames: int = 0,  # number of frames for latents; 0 denotes image
        sequence_parallel_size=1,
        num_block_chunks: int = 1,  # number of chunks for block
        gradient_checkpointing: Optional[str] = None,
        adaln_norm_type: str = "legacy",
        use_navit: bool = False,
        use_rope=False,
        enable_cached_compile=False,
        compile_mem_gc_threshold=-1,
        add_motion_score=False,
        condition_type="inpaint_mask",
        use_depth_cond=False,
    ):
        super().__init__()
        if sequence_parallel_size is None:
            sequence_parallel_size = 1  # adapt old models, remove in the future
        self.use_navit = use_navit
        self.use_rope = use_rope
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.enable_cached_compile = enable_cached_compile
        inner_dim = num_attention_heads * attention_head_dim
        if not isinstance(cross_attention_dim, list):
            cross_attention_dim = [cross_attention_dim, cross_attention_dim]

        assert (
            in_channels is not None and patch_size is not None
        ), "Only support self.is_input_patches for simplicity"
        assert norm_type in [
            "ada_norm_zero_nolabel",
            "ada_norm_zero",
        ], f"norm_type {norm_type} is not supported"

        self.add_motion_score = add_motion_score
        self.condition_type = condition_type
        if condition_type == "inpaint_mask":
            self.cond_conv = ConvitConditioningEmbedding(
                inner_dim,
                conditioning_channels=5 if use_depth_cond else 4,
                block_out_channels=(16, 32, 96, 256, 512),
            )
        else:
            self.cond_conv = None

        # 2. Define input layers
        assert (
            sample_size is not None
        ), "Transformer2DModel over patched input must provide sample_size"

        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        extra_kwargs = {}
        _patch_embed_class = PatchEmbed3D
        self.pos_embed = _patch_embed_class(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=1,
            **extra_kwargs,
        )

        if self.add_motion_score:
            self.time_embedder = CombinedTimestepGuidanceEmbedding(inner_dim)
        else:
            self.time_embedder = PlainTimestepEmbedding(None, inner_dim)

        # 3. Define transformers blocks
        if sequence_parallel_size is None:
            sequence_parallel_size = 0
        elif sequence_parallel_size == 1:
            print("\033[91m" + "=" * 50 + "\033[0m")
            print(
                "\033[91m"
                + "WARNING: `sequence_parallel_size=1` is not supported now. Cast to `sequence_parallel_size=0` that doesn't use sequence parallel."
                + "\033[0m"
            )
            print("\033[91m" + "=" * 50 + "\033[0m")
            sequence_parallel_size = 0
        assert sequence_parallel_size != 1, "FIXME: sequence_parallel_size 1 does not work now"
        block_extra_kwargs = dict(
            adaln_norm_type="layernorm" if adaln_norm_type == "legacy" else adaln_norm_type,
            use_navit=False,
        )

        if sequence_parallel_size > 1:
            block_extra_kwargs.update(
                {
                    "sequence_parallel_size": sequence_parallel_size,
                }
            )
            assert (
                self.num_attention_heads % sequence_parallel_size == 0
            ), f"num_heads {self.num_attention_heads} should be divisible by sequence_parallel_size {sequence_parallel_size}"

        if enable_cached_compile:
            block_extra_kwargs.update(
                {
                    "enable_cached_compile": enable_cached_compile,
                    "compile_mem_gc_threshold": compile_mem_gc_threshold,
                }
            )

        raw_transformer_blocks = [
            TransformerBlockGoku(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim[d % 2],
                activation_fn=activation_fn,
                num_embeds_ada_norm=num_embeds_ada_norm,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                attention_type=attention_type,
                **block_extra_kwargs,
            )
            for d in range(num_layers)
        ]

        self.num_block_chunks = num_block_chunks
        if num_block_chunks == 1:
            self.transformer_blocks = nn.ModuleList(raw_transformer_blocks)
        else:
            self.transformer_blocks = nn.ModuleList()
            assert (
                len(raw_transformer_blocks) % num_block_chunks == 0
            ), f"total number of blocks {len(raw_transformer_blocks)} should be divisible by num_blocks_in_a_chunk {num_block_chunks}"
            num_blocks_in_a_chunk = len(raw_transformer_blocks) // num_block_chunks
            for i in range(0, len(raw_transformer_blocks), num_blocks_in_a_chunk):
                self.transformer_blocks.append(
                    MultipleLayers(raw_transformer_blocks, num_blocks_in_a_chunk, i)
                )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out_2 = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)

        self.gradient_checkpointing = gradient_checkpointing

    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = diffusers.__version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            # make `sequence_parallel_group` happy
            elif isinstance(value, PosixPath) or isinstance(value, ProcessGroup):
                value = str(value)
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def post_setup_sequence_parallel(
        self, sequence_parallel_group, sequence_parallel_strategy="long-seq"
    ):
        self.sequence_parallel_group = sequence_parallel_group
        assert sequence_parallel_strategy in ["long-seq", "ulysses"]
        self.sequence_parallel_strategy = sequence_parallel_strategy
        if sequence_parallel_group is None:
            self.sequence_parallel_rank = None
        else:
            self.sequence_parallel_rank = dist.get_rank(sequence_parallel_group)

        for block in self.transformer_blocks:
            if isinstance(block, MultipleLayers):
                for m in block.module:
                    if hasattr(m, "post_setup_sequence_parallel"):
                        m.post_setup_sequence_parallel(
                            sequence_parallel_group, sequence_parallel_strategy
                        )
            else:
                if hasattr(block, "post_setup_sequence_parallel"):
                    block.post_setup_sequence_parallel(
                        sequence_parallel_group, sequence_parallel_strategy
                    )

    def _set_gradient_checkpointing(self, module, value=None):
        raise NotImplementedError("Double check here")
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def dit_initialize_weights(self):
        print("Initializing weights...")

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embedder.timestep_embedder.linear_1.weight, std=0.02)
        nn.init.normal_(self.time_embedder.timestep_embedder.linear_2.weight, std=0.02)

        if self.add_motion_score:
            nn.init.normal_(self.time_embedder.guidance_embedder.linear_1.weight, std=0.02)
            nn.init.normal_(self.time_embedder.guidance_embedder.linear_2.weight, std=0.02)

        def init_one_block(block):
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)

            if block.norm1.emb is None:
                return

            # Initialize timestep embedding MLP:
            nn.init.normal_(block.norm1.emb.timestep_embedder.linear_1.weight, std=0.02)
            nn.init.normal_(block.norm1.emb.timestep_embedder.linear_2.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for raw_block in self.transformer_blocks:
            if isinstance(raw_block, MultipleLayers):
                # chunk block
                for block in raw_block.module:
                    init_one_block(block)
            else:
                init_one_block(raw_block)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out_1.weight, 0)
        nn.init.constant_(self.proj_out_1.bias, 0)
        nn.init.constant_(self.proj_out_2.weight, 0)
        nn.init.constant_(self.proj_out_2.bias, 0)

    def embed_init(self):
        if self.num_embeds_ada_norm != 1000:
            # Initialize label embedding table:
            for block in self.transformer_blocks:
                # self.transformer_blocks.27.norm1.emb.class_embedder.embedding_table.weight
                nn.init.normal_(block.norm1.emb.class_embedder.embedding_table.weight, std=0.02)

    def get_fsdp_wrap_module_list(self):
        return self.transformer_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditional_pixel_values: torch.Tensor = None,
        conditional_masks: torch.Tensor = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        pos_embed=None,
        motion_score=None,
        desired_step_size: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        pooled_projections=None,
        frame_idxs=None,
        cumsum_q_len=None,
        cumsum_kv_len=None,
        batch_q_len=None,
        batch_kv_len=None,
        height=None,  # used for navit inference
        return_dict: bool = True,
        use_reentrant_checkpoint: bool = False,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.use_rope and pos_embed is None:
            num_latent_frames, latent_h, latent_w = hidden_states.shape[-3:]
            self.num_latent_frames = num_latent_frames
            seq_len = (latent_h // 2) * (latent_w // 2) * (self.num_latent_frames or 1)
            if self.use_rope:
                rotary_emb = prepare_rotary_positional_embeddings(
                    grid_h=latent_h // self.patch_size,
                    grid_w=latent_w // self.patch_size,
                    grid_t=num_latent_frames // self.patch_size_t,
                    attention_head_dim=self.attention_head_dim,
                    device=torch.device("cpu"),
                )
                pos_embed = torch.stack(rotary_emb, -1).float().to(hidden_states.device)
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        self.num_latent_frames = hidden_states.shape[2]
        if self.condition_type == "inpaint_mask" and self.cond_conv is not None:
            if conditional_masks is not None:
                conditional_pixel_values = torch.cat(
                    [conditional_pixel_values, conditional_masks], dim=1
                )
            conditional_inputs = self.cond_conv(conditional_pixel_values)
        else:
            if self.condition_type == "latent_concat" and self.cond_channels > 0:
                # print(hidden_states.shape, conditional_pixel_values.shape)
                hidden_states = torch.cat([hidden_states, conditional_pixel_values], dim=1)
            conditional_inputs = None

        rope_pos_embed = pos_embed if self.use_rope else None
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )

        hidden_states, latent_height, latent_width = self.pos_embed(
            hidden_states, conditional_inputs
        )

        if motion_score is not None:
            timestep = self.time_embedder(timestep, motion_score, hidden_dtype=hidden_states.dtype)
        else:
            timestep = self.time_embedder(timestep, None, hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        grad_scale = True  # debug NOTE
        # print("Before chunk: ", hidden_states.shape) # 16, 1024, 1152  B T D
        if self.config.sequence_parallel_size > 1:
            hidden_states = hidden_states.chunk(self.sequence_parallel_size, dim=1)[
                dist.get_rank(self.sequence_parallel_group)
            ]

        if isinstance(encoder_hidden_states, list):
            # hard-code, the first is CLIP-L embedding
            encoder_attention_mask = [None, encoder_attention_mask]
        else:
            encoder_hidden_states = [encoder_hidden_states, encoder_hidden_states]
            encoder_attention_mask = [encoder_attention_mask, encoder_attention_mask]

        for blk_idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                q_height=latent_height,
                q_width=latent_width,
                attention_mask=attention_mask,
                encoder_hidden_states=(
                    encoder_hidden_states
                    if self.num_block_chunks > 1
                    else encoder_hidden_states[blk_idx % 2]
                ),
                encoder_attention_mask=(
                    encoder_attention_mask
                    if self.num_block_chunks > 1
                    else encoder_attention_mask[blk_idx % 2]
                ),
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                rope_cache=None,
                added_cond_kwargs=added_cond_kwargs,
                checkpointing=self.gradient_checkpointing if self.training else None,
                sequence_parallel_rank=self.sequence_parallel_rank,
                sequence_parallel_group=None,
                sequence_parallel_strategy=self.sequence_parallel_strategy,
                cumsum_q_len=cumsum_q_len,
                cumsum_kv_len=cumsum_kv_len,
                batch_q_len=batch_q_len,
                batch_kv_len=batch_kv_len,
                rope_pos_embed=rope_pos_embed,
                time_pos_embed=None,
                use_reentrant_checkpoint=use_reentrant_checkpoint,
            )

        if self.config.sequence_parallel_size > 1:
            if grad_scale:
                hidden_states = gather_sequence(
                    hidden_states,
                    process_group=self.sequence_parallel_group,
                    dim=0 if self.use_navit else 1,
                    grad_scale="up",
                )
            else:
                hidden_states = gather_forward_split_backward(
                    hidden_states,
                    dim=0 if self.use_navit else 1,
                    process_group=self.sequence_parallel_group,
                )
            # NOTE(wuhui): We dont gather time_pos_embed here because no need for it in below codes
            # time_pos_embed = gather_forward_split_backward(hidden_states, dim=0 if self.use_navit else 1, process_group=self.sequence_parallel_group)

        # 3. Output
        shift, scale = self.proj_out_1(F.silu(timestep)).float().chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states).float() * (1 + scale[:, None]) + shift[:, None]
        ).to(hidden_states.dtype)
        hidden_states = self.proj_out_2(hidden_states)

        if self.num_latent_frames == 0:
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.patch_size,
                    width * self.patch_size,
                )
            )
        else:
            # TODO: double check video unpatchify
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    self.num_latent_frames,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nfhwpqc->ncfhpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    self.num_latent_frames,
                    height * self.patch_size,
                    width * self.patch_size,
                )
            )

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
