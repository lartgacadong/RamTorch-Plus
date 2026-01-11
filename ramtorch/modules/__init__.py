"""
RamTorch CPU-Bouncing Modules

Memory-efficient neural network modules that keep parameters on CPU
and transfer them to GPU on-demand using asynchronous CUDA/HIP streams.
"""

from .linear import Linear, CPUBouncingLinear
from .embedding import Embedding, CPUBouncingEmbedding
from .conv import Conv2d, Conv3d, CPUBouncingConv2d, CPUBouncingConv3d
from .norm import LayerNorm, RMSNorm, CPUBouncingLayerNorm, CPUBouncingRMSNorm

__all__ = [
    # Linear
    "Linear",
    "CPUBouncingLinear",
    # Embedding
    "Embedding",
    "CPUBouncingEmbedding",
    # Convolution
    "Conv2d",
    "Conv3d",
    "CPUBouncingConv2d",
    "CPUBouncingConv3d",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "CPUBouncingLayerNorm",
    "CPUBouncingRMSNorm",
]
