"""
CPU Embedding Module

A memory-efficient embedding layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA/HIP streams.

This approach interleaves compute and data transfer, making it useful for:
- Very large embedding tables that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import record_function
from ..accelerator import Accelerator

# --- Per-device global state registry ---
_DEVICE_STATE = {}


def _get_device_state(device=None):
    """Get or initialize per-device state."""
    accelerator = Accelerator.create(device)
    device_key = accelerator.key

    if device_key not in _DEVICE_STATE:
        with accelerator.device_context():
            _DEVICE_STATE[device_key] = {
                "accelerator": accelerator,
                # streams & events
                "transfer_stream": accelerator.new_stream(),
                "transfer_grad_stream": accelerator.new_stream(),
                "transfer_forward_finished_event": accelerator.new_event(),
                "compute_forward_start_event": accelerator.new_event(),
                "transfer_backward_finished_event": accelerator.new_event(),
                "compute_backward_start_event": accelerator.new_event(),
                "compute_backward_finished_event": accelerator.new_event(),
                "transfer_grad_finished_event": accelerator.new_event(),
                # buffers (ping-pong)
                "w_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device_key]


def _invoke_tensor_hooks(tensor, grad):
    """Invoke backward hooks registered on a tensor."""
    if hasattr(tensor, "_ramtorch_backward_hooks") and tensor._ramtorch_backward_hooks:
        for hook_id, hook_fn in tensor._ramtorch_backward_hooks.items():
            result = hook_fn(grad)
            if result is not None:
                grad = result
    return grad


def _invoke_post_accum_tensor_hooks(tensor):
    """Invoke post accumulate grad hooks registered on a tensor."""
    if (
        hasattr(tensor, "_ramtorch_post_accumulate_grad_hooks")
        and tensor._ramtorch_post_accumulate_grad_hooks
    ):
        for hook_id, hook_fn in tensor._ramtorch_post_accumulate_grad_hooks.items():
            hook_fn(tensor)


class BouncingEmbeddingFn(torch.autograd.Function):
    """
    Custom autograd function implementing the bouncing embedding operation.

    This function handles:
    1. Asynchronous transfer of embedding weights from CPU to GPU
    2. Proper synchronization between transfer and compute streams
    3. Gradient computation and accumulation back to CPU
    """

    @staticmethod
    def forward(ctx, indices, weight_cpu, device, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
        """
        Forward pass of bouncing embedding layer.

        Args:
            ctx: PyTorch autograd context
            indices: Input indices tensor on GPU
            weight_cpu: Embedding weight matrix stored on CPU
            device: Target GPU device for computation
            padding_idx: Index for padding token (gradient zeroed)
            max_norm: Max norm for renormalization
            norm_type: p-norm type for renormalization
            scale_grad_by_freq: Scale gradients by inverse frequency
            sparse: Use sparse gradients (not supported in bouncing mode)

        Returns:
            Embedded output tensor
        """
        state = _get_device_state(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        w_buffers = state["w_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        # Get index from clock and flip
        selected_buffer = state["forward_clk"]
        state["forward_clk"] ^= 1

        # Enqueue transfer on transfer stream
        with accelerator.use_stream(transfer_stream):
            transfer_stream.wait_event(compute_forward_start_event)

            with record_function("embedding_forward_weight_transfer"):
                w_buffers[selected_buffer] = weight_cpu.to(tensor_device, non_blocking=True)

            transfer_forward_finished_event.record()

        with record_function("embedding_forward_compute"):
            accelerator.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()

            out = F.embedding(
                indices,
                w_buffers[selected_buffer],
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=False,  # Always dense for bouncing
            )

        # Save for backward
        ctx.save_for_backward(indices, weight_cpu)
        ctx.device = device
        ctx.padding_idx = padding_idx
        ctx.num_embeddings = weight_cpu.shape[0]
        ctx.scale_grad_by_freq = scale_grad_by_freq
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for gradient computation.

        Embedding gradient is computed by scattering grad_out to the weight positions
        indicated by the saved indices.
        """
        indices, weight_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_grad_buffers = state["w_grad_buffers"]
        w_grad_accum_buffers = state["w_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]
        transfer_grad_finished_event = state["transfer_grad_finished_event"]

        # Get index from clock and flip
        selected_buffer = state["backward_clk"]
        state["backward_clk"] ^= 1

        # Transfer existing gradients for accumulation
        with accelerator.use_stream(transfer_stream):
            with record_function("embedding_backward_grad_accumulator_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)

                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(tensor_device, non_blocking=True)
                    if weight_cpu.grad is not None
                    else None
                )

                transfer_backward_finished_event.record()

        accelerator.current_stream().wait_event(transfer_backward_finished_event)
        compute_backward_start_event.record()

        with record_function("embedding_backward_compute"):
            # Compute embedding gradient via scatter
            # grad_weight has shape [num_embeddings, embedding_dim]
            grad_weight = torch.zeros(
                ctx.num_embeddings,
                grad_out.shape[-1],
                dtype=grad_out.dtype,
                device=tensor_device,
            )

            # Flatten indices and grad_out for index_add_
            flat_indices = indices.flatten()
            flat_grad_out = grad_out.flatten(0, -2)  # [N, embedding_dim]

            # Scale by inverse frequency if requested
            if ctx.scale_grad_by_freq:
                counts = torch.bincount(flat_indices, minlength=ctx.num_embeddings).float()
                counts = counts.clamp(min=1.0)
                scale = 1.0 / counts[flat_indices]
                flat_grad_out = flat_grad_out * scale.unsqueeze(-1)

            grad_weight.index_add_(0, flat_indices, flat_grad_out)

            # Zero out padding_idx gradient
            if ctx.padding_idx is not None:
                grad_weight[ctx.padding_idx] = 0

            w_grad_buffers[selected_buffer] = grad_weight

            # Gradient accumulation
            with record_function("embedding_backward_weight_grad_accumulate"):
                w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                    weight_cpu, w_grad_buffers[selected_buffer]
                )

                if w_grad_accum_buffers[selected_buffer] is not None:
                    w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU
        with record_function("embedding_backward_grad_transfer"):
            with accelerator.use_stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)
                weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)
                transfer_grad_finished_event.record()

        # Return gradients: (indices, weight, device, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
        return None, None, None, None, None, None, None, None


class CPUBouncingEmbedding(nn.Module):
    """
    Embedding layer with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.Embedding but with different
    memory characteristics:
    - Parameters stored on CPU (using pinned memory for fast transfers)
    - Transferred to GPU only during forward/backward passes
    - Automatic cleanup after each operation

    Trade-offs:
    + Drastically reduced GPU memory usage for large vocabularies
    + Enables training with much larger embedding tables
    - Requires batching to mask the latency
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device="cuda",
        dtype=None,
        _weight=None,
    ):
        """
        Initialize CPU embedding layer.

        Args:
            num_embeddings: Size of the embedding dictionary
            embedding_dim: Size of each embedding vector
            padding_idx: Index for padding token (gradient zeroed)
            max_norm: Max norm for renormalization
            norm_type: p-norm type for renormalization
            scale_grad_by_freq: Scale gradients by inverse frequency
            sparse: Ignored (always dense for bouncing)
            device: Target GPU device for computation
            dtype: Data type for embeddings
            _weight: Optional pre-initialized weight tensor
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = False  # Always dense for bouncing mode

        if device is None:
            device = "cuda"
        self.device = device

        if dtype is None:
            dtype = torch.float32

        # Parameters live on CPU
        if _weight is not None:
            self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
        else:
            self.weight = nn.Parameter(
                torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu").pin_memory()
            )
            nn.init.normal_(self.weight)

        # Mark as ramtorch tensor
        self.weight.is_ramtorch = True
        self.is_ramtorch = True

        # Handle padding_idx initialization
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
            self.weight.data = self.weight.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        """No-op: weights stay on CPU."""
        return self

    def cpu(self):
        """No-op: weights already on CPU."""
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CPU embedding layer.

        Args:
            input_ids: Input indices tensor (should be on GPU)

        Returns:
            Embedded output tensor
        """
        return BouncingEmbeddingFn.apply(
            input_ids,
            self.weight,
            self.device,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        return s.format(**self.__dict__)


# Alias for convenience
Embedding = CPUBouncingEmbedding
