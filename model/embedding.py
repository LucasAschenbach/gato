"""
Module description:
-------------------
This module defines the Embedding class and its subclasses for different modalities.

Classes:
    - StructuredDataEmbedding: Lookup table based embedding for structured data.
    - ImageEmbedding: ResNet based embedding for images.
"""

import torch
import torch.nn as nn
from transformers import ResNetModel

class Embedding(nn.Module):
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")


class StructuredDataEmbedding(Embedding):
    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__(embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = x.long()
        x = self.embedding(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, num_group_norm_groups: int, num_channels: int):
        super().__init__()
        self.num_group_norm_groups = num_group_norm_groups
        self.num_channels = num_channels

        self.conv_proj = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=2, padding=0)
        self.gn_proj = nn.GroupNorm(num_group_norm_groups, num_channels)

        self.model = nn.Sequential(
            nn.GroupNorm(num_group_norm_groups, num_channels // 2),
            nn.GELU(),
            nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_group_norm_groups, num_channels),
            nn.GELU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        residual = self.conv_proj(self.gn_proj(x))
        x = self.model(x) + residual
        return x


class ImageEmbedding(Embedding):
    def __init__(self, embedding_dim: int, num_group_norm_groups: int, layer_width: int):
        super().__init__(embedding_dim)
        self.num_group_norm_groups = num_group_norm_groups
        self.layer_width = layer_width

        input_channels = 3
        root_channels = 96
        self.resnet_root = nn.Sequential(
            nn.Conv2d(input_channels, root_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_group_norm_groups, root_channels),
            nn.GELU(),
        )

        self.resnet_blocks = [
            ResidualBlock(num_group_norm_groups, root_channels * 2),
            ResidualBlock(num_group_norm_groups, root_channels * 4),
            ResidualBlock(num_group_norm_groups, root_channels * 8), # == embedding_dim
        ]

    def forward(self, x):
        x = self.resnet_root(x)
        for block in self.resnet_blocks:
            x = block(x)
        return x.reshape(-1, x.shape[1], self.layer_width)
