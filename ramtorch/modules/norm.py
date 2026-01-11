"""
CPU Normalization Modules

Memory-efficient LayerNorm and RMSNorm implementations that keep parameters on CPU
and transfer them to GPU on-demand using asynchronous CUDA/HIP streams.

This approach interleaves compute and data transfer, making it useful for:
- Very large models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List

from torch.profiler import record_function
from ..accelerator import Accelerator

# --- Per-device global state registry ---
_DEVICE_STATE_LAYERNORM = {}
_DEVICE_STATE_RMSNORM = {}


def _get_device_state_layernorm(device=None):
    """Get or initialize per-device state for LayerNorm."""
    accelerator = Accelerator.create(device)
    device_key = accelerator.key

    if device_key not in _DEVICE_STATE_LAYERNORM:
        with accelerator.device_context():
            _DEVICE_STATE_LAYERNORM[device_key] = {
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
                "b_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE_LAYERNORM[device_key]


def _get_device_state_rmsnorm(device=None):
    """Get or initialize per-device state for RMSNorm."""
    accelerator = Accelerator.create(device)
    device_key = accelerator.key

    if device_key not in _DEVICE_STATE_RMSNORM:
        with accelerator.device_context():
            _DEVICE_STATE_RMSNORM[device_key] = {
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
                "b_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE_RMSNORM[device_key]


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


def _normalize_shape(normalized_shape) -> Tuple[int, ...]:
    """Normalize shape to tuple."""
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)


class BouncingLayerNormFn(torch.autograd.Function):
    """Custom autograd function for CPU-bouncing LayerNorm."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device, normalized_shape, eps):
        state = _get_device_state_layernorm(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        w_buffers = state["w_buffers"]
        b_buffers = state["b_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        # Get index from clock and flip
        selected_buffer = state["forward_clk"]
        state["forward_clk"] ^= 1

        # Enqueue transfer on transfer stream
        with accelerator.use_stream(transfer_stream):
            transfer_stream.wait_event(compute_forward_start_event)

            with record_function("layernorm_forward_weight_bias_transfer"):
                w_buffers[selected_buffer] = (
                    weight_cpu.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None
                    else None
                )
                b_buffers[selected_buffer] = (
                    bias_cpu.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None
                    else None
                )

            transfer_forward_finished_event.record()

        with record_function("layernorm_forward_compute"):
            accelerator.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()

            out = F.layer_norm(
                x,
                normalized_shape,
                w_buffers[selected_buffer],
                b_buffers[selected_buffer],
                eps,
            )

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        state = _get_device_state_layernorm(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_buffers = state["w_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]
        w_grad_accum_buffers = state["w_grad_accum_buffers"]
        b_grad_accum_buffers = state["b_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]
        transfer_grad_finished_event = state["transfer_grad_finished_event"]

        # Get index from clock and flip
        selected_buffer = state["backward_clk"]
        state["backward_clk"] ^= 1

        # Transfer weights and existing gradients
        with accelerator.use_stream(transfer_stream):
            with record_function("layernorm_backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_buffers[selected_buffer] = (
                    weight_cpu.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None
                    else None
                )

            with record_function("layernorm_backward_grad_accumulator_transfer"):
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None and weight_cpu.grad is not None
                    else None
                )
                b_grad_accum_buffers[selected_buffer] = (
                    bias_cpu.grad.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None and bias_cpu.grad is not None
                    else None
                )
                transfer_backward_finished_event.record()

        accelerator.current_stream().wait_event(transfer_backward_finished_event)
        compute_backward_start_event.record()

        with record_function("layernorm_backward_compute"):
            # Compute layer norm backward manually
            # Forward was: out = (x - mean) / std * weight + bias
            # where std = sqrt(var + eps)

            # Dimensions to normalize over
            dims = tuple(range(-len(normalized_shape), 0))

            # Recompute forward pass statistics
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            std = (var + eps).sqrt()
            x_norm = (x - mean) / std

            # Number of elements in normalized dimensions
            n = 1
            for d in normalized_shape:
                n *= d

            # Grad w.r.t. weight: sum over batch dims of grad_out * x_norm
            if weight_cpu is not None:
                batch_dims = tuple(range(x.ndim - len(normalized_shape)))
                w_grad_buffers[selected_buffer] = (grad_out * x_norm).sum(dim=batch_dims)

                with record_function("layernorm_backward_weight_grad_accumulate"):
                    w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        weight_cpu, w_grad_buffers[selected_buffer]
                    )

                    if w_grad_accum_buffers[selected_buffer] is not None:
                        w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                    weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(weight_cpu)
                    del weight_cpu.ramtorch_grad

            # Grad w.r.t. bias: sum over batch dims
            if bias_cpu is not None:
                batch_dims = tuple(range(x.ndim - len(normalized_shape)))
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=batch_dims)

                with record_function("layernorm_backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        bias_cpu, b_grad_buffers[selected_buffer]
                    )

                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]

                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad

            # Grad w.r.t. input
            if w_buffers[selected_buffer] is not None:
                grad_x_norm = grad_out * w_buffers[selected_buffer]
            else:
                grad_x_norm = grad_out

            # LayerNorm backward formula
            grad_var = (-0.5 * grad_x_norm * (x - mean) / (var + eps).pow(1.5)).sum(dim=dims, keepdim=True)
            grad_mean = (-grad_x_norm / std).sum(dim=dims, keepdim=True) + grad_var * (-2.0 * (x - mean)).mean(dim=dims, keepdim=True)
            grad_input = grad_x_norm / std + grad_var * 2.0 * (x - mean) / n + grad_mean / n

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU
        with record_function("layernorm_backward_grad_transfer"):
            with accelerator.use_stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)

                if weight_cpu is not None:
                    weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                if bias_cpu is not None:
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                transfer_grad_finished_event.record()

        # Return gradients for all inputs
        return grad_input, None, None, None, None, None


class BouncingRMSNormFn(torch.autograd.Function):
    """Custom autograd function for CPU-bouncing RMSNorm."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device, normalized_shape, eps, use_weight_addition):
        state = _get_device_state_rmsnorm(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        w_buffers = state["w_buffers"]
        b_buffers = state["b_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        # Get index from clock and flip
        selected_buffer = state["forward_clk"]
        state["forward_clk"] ^= 1

        # Enqueue transfer on transfer stream
        with accelerator.use_stream(transfer_stream):
            transfer_stream.wait_event(compute_forward_start_event)

            with record_function("rmsnorm_forward_weight_bias_transfer"):
                w_buffers[selected_buffer] = (
                    weight_cpu.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None
                    else None
                )
                b_buffers[selected_buffer] = (
                    bias_cpu.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None
                    else None
                )

            transfer_forward_finished_event.record()

        with record_function("rmsnorm_forward_compute"):
            accelerator.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()

            # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
            dims = tuple(range(-len(normalized_shape), 0))
            variance = x.float().pow(2).mean(dim=dims, keepdim=True)
            x_norm = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)

            if w_buffers[selected_buffer] is not None:
                if use_weight_addition:
                    # Gemma-style: x_norm * (1 + weight)
                    out = x_norm * (1.0 + w_buffers[selected_buffer].float())
                else:
                    out = x_norm * w_buffers[selected_buffer]
            else:
                out = x_norm

            if b_buffers[selected_buffer] is not None:
                out = out + b_buffers[selected_buffer]

            out = out.to(dtype=x.dtype)

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.use_weight_addition = use_weight_addition
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps
        use_weight_addition = ctx.use_weight_addition

        state = _get_device_state_rmsnorm(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_buffers = state["w_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]
        w_grad_accum_buffers = state["w_grad_accum_buffers"]
        b_grad_accum_buffers = state["b_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]
        transfer_grad_finished_event = state["transfer_grad_finished_event"]

        # Get index from clock and flip
        selected_buffer = state["backward_clk"]
        state["backward_clk"] ^= 1

        # Transfer weights and existing gradients
        with accelerator.use_stream(transfer_stream):
            with record_function("rmsnorm_backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_buffers[selected_buffer] = (
                    weight_cpu.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None
                    else None
                )

            with record_function("rmsnorm_backward_grad_accumulator_transfer"):
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(tensor_device, non_blocking=True)
                    if weight_cpu is not None and weight_cpu.grad is not None
                    else None
                )
                b_grad_accum_buffers[selected_buffer] = (
                    bias_cpu.grad.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None and bias_cpu.grad is not None
                    else None
                )
                transfer_backward_finished_event.record()

        accelerator.current_stream().wait_event(transfer_backward_finished_event)
        compute_backward_start_event.record()

        with record_function("rmsnorm_backward_compute"):
            # Recompute forward pass
            dims = tuple(range(-len(normalized_shape), 0))
            x_float = x.float()
            variance = x_float.pow(2).mean(dim=dims, keepdim=True)
            rsqrt_var = torch.rsqrt(variance + eps)
            x_norm = x_float * rsqrt_var

            # Number of elements in normalized dimensions
            n = 1
            for d in normalized_shape:
                n *= d

            # Grad w.r.t. weight
            if weight_cpu is not None:
                batch_dims = tuple(range(x.ndim - len(normalized_shape)))

                if use_weight_addition:
                    # d/dw of x_norm * (1 + w) = x_norm
                    w_grad_buffers[selected_buffer] = (grad_out.float() * x_norm).sum(dim=batch_dims).to(weight_cpu.dtype)
                else:
                    w_grad_buffers[selected_buffer] = (grad_out.float() * x_norm).sum(dim=batch_dims).to(weight_cpu.dtype)

                with record_function("rmsnorm_backward_weight_grad_accumulate"):
                    w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        weight_cpu, w_grad_buffers[selected_buffer]
                    )

                    if w_grad_accum_buffers[selected_buffer] is not None:
                        w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                    weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(weight_cpu)
                    del weight_cpu.ramtorch_grad

            # Grad w.r.t. bias
            if bias_cpu is not None:
                batch_dims = tuple(range(x.ndim - len(normalized_shape)))
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=batch_dims).to(bias_cpu.dtype)

                with record_function("rmsnorm_backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        bias_cpu, b_grad_buffers[selected_buffer]
                    )

                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]

                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad

            # Grad w.r.t. input
            grad_out_float = grad_out.float()

            if w_buffers[selected_buffer] is not None:
                if use_weight_addition:
                    grad_x_norm = grad_out_float * (1.0 + w_buffers[selected_buffer].float())
                else:
                    grad_x_norm = grad_out_float * w_buffers[selected_buffer].float()
            else:
                grad_x_norm = grad_out_float

            # RMSNorm backward: d/dx of x * rsqrt(mean(x^2) + eps)
            # = rsqrt(var) - x * x * rsqrt(var)^3 / n
            grad_var = (-0.5 * grad_x_norm * x_float * rsqrt_var.pow(3)).sum(dim=dims, keepdim=True)
            grad_input = grad_x_norm * rsqrt_var + grad_var * 2.0 * x_float / n
            grad_input = grad_input.to(dtype=x.dtype)

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU
        with record_function("rmsnorm_backward_grad_transfer"):
            with accelerator.use_stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)

                if weight_cpu is not None:
                    weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                if bias_cpu is not None:
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                transfer_grad_finished_event.record()

        # Return gradients for all inputs
        return grad_input, None, None, None, None, None, None


class CPUBouncingLayerNorm(nn.Module):
    """
    LayerNorm with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.LayerNorm but with different
    memory characteristics:
    - Parameters stored on CPU (using pinned memory for fast transfers)
    - Transferred to GPU only during forward/backward passes
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        self.normalized_shape = _normalize_shape(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if device is None:
            device = "cuda"
        self.device = device

        if dtype is None:
            dtype = torch.float32

        if elementwise_affine:
            if _weight is not None:
                self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
            else:
                self.weight = nn.Parameter(
                    torch.ones(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
                )
            self.weight.is_ramtorch = True

            if bias:
                if _bias is not None:
                    self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
                else:
                    self.bias = nn.Parameter(
                        torch.zeros(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
                    )
                self.bias.is_ramtorch = True
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.is_ramtorch = True

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=torch.float32)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        """No-op: weights stay on CPU."""
        return self

    def cpu(self):
        """No-op: weights already on CPU."""
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, None, None, self.eps)

        return BouncingLayerNormFn.apply(
            x,
            self.weight,
            self.bias,
            self.device,
            self.normalized_shape,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


class CPUBouncingRMSNorm(nn.Module):
    """
    RMSNorm with CPU-stored parameters that bounce to GPU on demand.

    RMSNorm normalizes by the root mean square of the input, without centering.
    Supports Gemma-style weight addition where weight is initialized to 0 and
    applied as x * (1 + weight) instead of x * weight.

    This module provides a drop-in replacement for various RMSNorm implementations.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
        use_weight_addition: bool = False,
    ):
        """
        Initialize RMSNorm.

        Args:
            normalized_shape: Shape of the normalized dimensions
            eps: Epsilon for numerical stability
            elementwise_affine: Whether to use learnable weight/bias
            bias: Whether to include bias (rare for RMSNorm)
            device: Target GPU device for computation
            dtype: Data type for parameters
            _weight: Optional pre-initialized weight tensor
            _bias: Optional pre-initialized bias tensor
            use_weight_addition: If True, apply as x * (1 + w) instead of x * w
                                (Gemma-style initialization)
        """
        super().__init__()
        self.normalized_shape = _normalize_shape(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_weight_addition = use_weight_addition

        if device is None:
            device = "cuda"
        self.device = device

        if dtype is None:
            dtype = torch.float32

        if elementwise_affine:
            if _weight is not None:
                self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
            else:
                # Initialize to 0 if using weight addition (Gemma-style), else 1
                init_fn = torch.zeros if use_weight_addition else torch.ones
                self.weight = nn.Parameter(
                    init_fn(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
                )
            self.weight.is_ramtorch = True

            if bias:
                if _bias is not None:
                    self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
                else:
                    self.bias = nn.Parameter(
                        torch.zeros(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
                    )
                self.bias.is_ramtorch = True
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.is_ramtorch = True

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=torch.float32)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        """No-op: weights stay on CPU."""
        return self

    def cpu(self):
        """No-op: weights already on CPU."""
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            # No learnable parameters, just normalize
            dims = tuple(range(-len(self.normalized_shape), 0))
            variance = x.float().pow(2).mean(dim=dims, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(dtype=x.dtype)

        return BouncingRMSNormFn.apply(
            x,
            self.weight,
            self.bias,
            self.device,
            self.normalized_shape,
            self.eps,
            self.use_weight_addition,
        )

    def extra_repr(self) -> str:
        s = "{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}"
        if self.use_weight_addition:
            s += ", use_weight_addition=True"
        return s.format(**self.__dict__)


# Aliases for convenience
LayerNorm = CPUBouncingLayerNorm
RMSNorm = CPUBouncingRMSNorm
