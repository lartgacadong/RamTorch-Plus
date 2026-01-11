"""
CPU Convolution Modules

Memory-efficient Conv2d and Conv3d layer implementations that keep parameters on CPU
and transfer them to GPU on-demand using asynchronous CUDA/HIP streams.

This approach interleaves compute and data transfer, making it useful for:
- Very large convolutional models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from torch.profiler import record_function
from ..accelerator import Accelerator

# --- Per-device global state registry ---
_DEVICE_STATE_CONV2D = {}
_DEVICE_STATE_CONV3D = {}


def _get_device_state_conv2d(device=None):
    """Get or initialize per-device state for Conv2d."""
    accelerator = Accelerator.create(device)
    device_key = accelerator.key

    if device_key not in _DEVICE_STATE_CONV2D:
        with accelerator.device_context():
            _DEVICE_STATE_CONV2D[device_key] = {
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
                "w_bwd_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE_CONV2D[device_key]


def _get_device_state_conv3d(device=None):
    """Get or initialize per-device state for Conv3d."""
    accelerator = Accelerator.create(device)
    device_key = accelerator.key

    if device_key not in _DEVICE_STATE_CONV3D:
        with accelerator.device_context():
            _DEVICE_STATE_CONV3D[device_key] = {
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
                "w_bwd_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE_CONV3D[device_key]


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


def _normalize_tuple(value, n, name):
    """Normalize a value to an n-tuple."""
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (tuple, list)):
        if len(value) == n:
            return tuple(value)
        raise ValueError(f"{name} must have {n} elements, got {len(value)}")
    raise TypeError(f"{name} must be int or tuple, got {type(value)}")


class BouncingConv2dFn(torch.autograd.Function):
    """Custom autograd function for CPU-bouncing Conv2d."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device, stride, padding, dilation, groups, padding_mode):
        state = _get_device_state_conv2d(device)
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

            with record_function("conv2d_forward_weight_bias_transfer"):
                w_buffers[selected_buffer] = weight_cpu.to(tensor_device, non_blocking=True)
                b_buffers[selected_buffer] = (
                    bias_cpu.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None
                    else None
                )

            transfer_forward_finished_event.record()

        with record_function("conv2d_forward_compute"):
            accelerator.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()

            # Handle padding_mode
            if padding_mode != "zeros":
                # Calculate padding for F.pad
                pad_h, pad_w = padding
                x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
                out = F.conv2d(
                    x_padded,
                    w_buffers[selected_buffer],
                    b_buffers[selected_buffer],
                    stride,
                    (0, 0),  # No additional padding
                    dilation,
                    groups,
                )
            else:
                out = F.conv2d(
                    x,
                    w_buffers[selected_buffer],
                    b_buffers[selected_buffer],
                    stride,
                    padding,
                    dilation,
                    groups,
                )

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding_mode = padding_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state_conv2d(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers = state["w_bwd_buffers"]
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
            with record_function("conv2d_backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_bwd_buffers[selected_buffer] = weight_cpu.to(tensor_device, non_blocking=True)

            with record_function("conv2d_backward_grad_accumulator_transfer"):
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(tensor_device, non_blocking=True)
                    if weight_cpu.grad is not None
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

        with record_function("conv2d_backward_compute"):
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            padding_mode = ctx.padding_mode

            # Handle padding_mode for backward
            if padding_mode != "zeros":
                pad_h, pad_w = padding
                x_for_grad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=padding_mode)
                padding_for_conv = (0, 0)
            else:
                x_for_grad = x
                padding_for_conv = padding

            # Compute grad_input using conv_transpose2d
            grad_input = torch.nn.grad.conv2d_input(
                x.shape,
                w_bwd_buffers[selected_buffer],
                grad_out,
                stride,
                padding,
                dilation,
                groups,
            )

            # Compute grad_weight
            w_grad_buffers[selected_buffer] = torch.nn.grad.conv2d_weight(
                x_for_grad,
                weight_cpu.shape,
                grad_out,
                stride,
                padding_for_conv,
                dilation,
                groups,
            )

            # Weight gradient accumulation
            with record_function("conv2d_backward_weight_grad_accumulate"):
                w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                    weight_cpu, w_grad_buffers[selected_buffer]
                )

                if w_grad_accum_buffers[selected_buffer] is not None:
                    w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad

            # Bias gradient
            if bias_cpu is not None:
                # Sum over all dimensions except output channels
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=(0, 2, 3))

                with record_function("conv2d_backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        bias_cpu, b_grad_buffers[selected_buffer]
                    )

                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]

                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU
        with record_function("conv2d_backward_grad_transfer"):
            with accelerator.use_stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)
                weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                if bias_cpu is not None:
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                transfer_grad_finished_event.record()

        # Return gradients for all inputs
        return grad_input, None, None, None, None, None, None, None, None


class BouncingConv3dFn(torch.autograd.Function):
    """Custom autograd function for CPU-bouncing Conv3d."""

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device, stride, padding, dilation, groups, padding_mode):
        state = _get_device_state_conv3d(device)
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

            with record_function("conv3d_forward_weight_bias_transfer"):
                w_buffers[selected_buffer] = weight_cpu.to(tensor_device, non_blocking=True)
                b_buffers[selected_buffer] = (
                    bias_cpu.to(tensor_device, non_blocking=True)
                    if bias_cpu is not None
                    else None
                )

            transfer_forward_finished_event.record()

        with record_function("conv3d_forward_compute"):
            accelerator.current_stream().wait_event(transfer_forward_finished_event)
            compute_forward_start_event.record()

            # Handle padding_mode
            if padding_mode != "zeros":
                # Calculate padding for F.pad
                pad_d, pad_h, pad_w = padding
                x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode=padding_mode)
                out = F.conv3d(
                    x_padded,
                    w_buffers[selected_buffer],
                    b_buffers[selected_buffer],
                    stride,
                    (0, 0, 0),  # No additional padding
                    dilation,
                    groups,
                )
            else:
                out = F.conv3d(
                    x,
                    w_buffers[selected_buffer],
                    b_buffers[selected_buffer],
                    stride,
                    padding,
                    dilation,
                    groups,
                )

        # Save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding_mode = padding_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state_conv3d(device)
        accelerator = state["accelerator"]
        tensor_device = accelerator.tensor_device
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers = state["w_bwd_buffers"]
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
            with record_function("conv3d_backward_weight_transfer"):
                transfer_stream.wait_event(compute_backward_start_event)
                w_bwd_buffers[selected_buffer] = weight_cpu.to(tensor_device, non_blocking=True)

            with record_function("conv3d_backward_grad_accumulator_transfer"):
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(tensor_device, non_blocking=True)
                    if weight_cpu.grad is not None
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

        with record_function("conv3d_backward_compute"):
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            padding_mode = ctx.padding_mode

            # Handle padding_mode for backward
            if padding_mode != "zeros":
                pad_d, pad_h, pad_w = padding
                x_for_grad = F.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode=padding_mode)
                padding_for_conv = (0, 0, 0)
            else:
                x_for_grad = x
                padding_for_conv = padding

            # Compute grad_input
            grad_input = torch.nn.grad.conv3d_input(
                x.shape,
                w_bwd_buffers[selected_buffer],
                grad_out,
                stride,
                padding,
                dilation,
                groups,
            )

            # Compute grad_weight
            w_grad_buffers[selected_buffer] = torch.nn.grad.conv3d_weight(
                x_for_grad,
                weight_cpu.shape,
                grad_out,
                stride,
                padding_for_conv,
                dilation,
                groups,
            )

            # Weight gradient accumulation
            with record_function("conv3d_backward_weight_grad_accumulate"):
                w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                    weight_cpu, w_grad_buffers[selected_buffer]
                )

                if w_grad_accum_buffers[selected_buffer] is not None:
                    w_grad_buffers[selected_buffer] += w_grad_accum_buffers[selected_buffer]

                weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad

            # Bias gradient
            if bias_cpu is not None:
                # Sum over all dimensions except output channels
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=(0, 2, 3, 4))

                with record_function("conv3d_backward_bias_grad_accumulate"):
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        bias_cpu, b_grad_buffers[selected_buffer]
                    )

                    if b_grad_accum_buffers[selected_buffer] is not None:
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[selected_buffer]

                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad

            compute_backward_finished_event.record()

        # Transfer gradients back to CPU
        with record_function("conv3d_backward_grad_transfer"):
            with accelerator.use_stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)
                weight_cpu.grad = w_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                if bias_cpu is not None:
                    bias_cpu.grad = b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)

                transfer_grad_finished_event.record()

        # Return gradients for all inputs
        return grad_input, None, None, None, None, None, None, None, None


class CPUBouncingConv2d(nn.Module):
    """
    Conv2d layer with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.Conv2d but with different
    memory characteristics:
    - Parameters stored on CPU (using pinned memory for fast transfers)
    - Transferred to GPU only during forward/backward passes
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(kernel_size, 2, "kernel_size")
        self.stride = _normalize_tuple(stride, 2, "stride")
        self.padding = _normalize_tuple(padding, 2, "padding")
        self.dilation = _normalize_tuple(dilation, 2, "dilation")
        self.groups = groups
        self.padding_mode = padding_mode

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
                torch.empty(
                    out_channels,
                    in_channels // groups,
                    *self.kernel_size,
                    dtype=dtype,
                    device="cpu",
                ).pin_memory()
            )
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.weight.is_ramtorch = True

        if bias:
            if _bias is not None:
                self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
            else:
                self.bias = nn.Parameter(
                    torch.zeros(out_channels, dtype=dtype, device="cpu").pin_memory()
                )
            self.bias.is_ramtorch = True
        else:
            self.register_parameter("bias", None)

        self.is_ramtorch = True

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
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
        return BouncingConv2dFn.apply(
            x,
            self.weight,
            self.bias,
            self.device,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.padding_mode,
        )

    def extra_repr(self) -> str:
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class CPUBouncingConv3d(nn.Module):
    """
    Conv3d layer with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.Conv3d but with different
    memory characteristics:
    - Parameters stored on CPU (using pinned memory for fast transfers)
    - Transferred to GPU only during forward/backward passes
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(kernel_size, 3, "kernel_size")
        self.stride = _normalize_tuple(stride, 3, "stride")
        self.padding = _normalize_tuple(padding, 3, "padding")
        self.dilation = _normalize_tuple(dilation, 3, "dilation")
        self.groups = groups
        self.padding_mode = padding_mode

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
                torch.empty(
                    out_channels,
                    in_channels // groups,
                    *self.kernel_size,
                    dtype=dtype,
                    device="cpu",
                ).pin_memory()
            )
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.weight.is_ramtorch = True

        if bias:
            if _bias is not None:
                self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
            else:
                self.bias = nn.Parameter(
                    torch.zeros(out_channels, dtype=dtype, device="cpu").pin_memory()
                )
            self.bias.is_ramtorch = True
        else:
            self.register_parameter("bias", None)

        self.is_ramtorch = True

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
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
        return BouncingConv3dFn.apply(
            x,
            self.weight,
            self.bias,
            self.device,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.padding_mode,
        )

    def extra_repr(self) -> str:
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


# Aliases for convenience
Conv2d = CPUBouncingConv2d
Conv3d = CPUBouncingConv3d
