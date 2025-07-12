
# Copyright 2024-2025 The GoKu Team Authors. All rights reserved.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvitConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires lightweight adapters to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):  # 0,1,2,3
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            if i % 2 == 0:
                self.blocks.append(
                    nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
                )
            else:
                self.blocks.append(
                    nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
                )

        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1
        )

    def forward(self, conditioning):
        num_frames, height, width = conditioning.shape[-3:]
        conditioning = rearrange(conditioning, "b c f h w -> (b f) c h w")

        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            if isinstance(block, nn.Conv3d):
                embedding = rearrange(embedding, "(b f) c h w -> b c f h w", f=num_frames)
                embedding = block(embedding)
                num_frames = embedding.shape[2]
                embedding = rearrange(embedding, "b c f h w -> (b f) c h w")
            else:
                embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        embedding = rearrange(embedding, "(b f) c h w -> b c f h w", f=num_frames)

        return embedding