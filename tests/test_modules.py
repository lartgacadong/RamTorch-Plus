"""
Tests for RamTorch CPU-bouncing modules.

These tests verify that the CPU-bouncing implementations produce correct results
and properly handle gradients.
"""

import unittest
import torch
import torch.nn as nn


def requires_cuda(fn):
    """Skip test if CUDA is not available."""
    return unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")(fn)


class TestEmbedding(unittest.TestCase):
    """Tests for CPUBouncingEmbedding."""

    @requires_cuda
    def test_forward_matches_nn_embedding(self):
        """Test that forward pass matches nn.Embedding."""
        from ramtorch.modules.embedding import Embedding

        vocab_size, embed_dim = 100, 64
        batch_size, seq_len = 4, 10

        # Create both versions with same weights
        torch.manual_seed(42)
        nn_emb = nn.Embedding(vocab_size, embed_dim)

        ramtorch_emb = Embedding(
            vocab_size, embed_dim, device="cuda", _weight=nn_emb.weight.data.clone()
        )
        nn_emb = nn_emb.cuda()

        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        # Forward pass
        nn_out = nn_emb(input_ids)
        ramtorch_out = ramtorch_emb(input_ids)

        self.assertTrue(torch.allclose(nn_out, ramtorch_out, atol=1e-5))

    @requires_cuda
    def test_backward_gradient_accumulation(self):
        """Test that gradients accumulate correctly."""
        from ramtorch.modules.embedding import Embedding

        vocab_size, embed_dim = 50, 32

        emb = Embedding(vocab_size, embed_dim, device="cuda")
        input_ids = torch.randint(0, vocab_size, (2, 5), device="cuda")

        # First backward pass
        out1 = emb(input_ids).sum()
        out1.backward()
        grad1 = emb.weight.grad.clone()

        # Second backward pass (should accumulate)
        out2 = emb(input_ids).sum()
        out2.backward()

        # Gradient should be roughly 2x (same input)
        self.assertTrue(torch.allclose(emb.weight.grad, grad1 * 2, atol=1e-5))

    @requires_cuda
    def test_padding_idx(self):
        """Test that padding_idx gradient is zeroed."""
        from ramtorch.modules.embedding import Embedding

        vocab_size, embed_dim = 50, 32
        padding_idx = 0

        emb = Embedding(vocab_size, embed_dim, padding_idx=padding_idx, device="cuda")

        # Include padding token in input
        input_ids = torch.tensor([[padding_idx, 1, 2, 3]], device="cuda")

        out = emb(input_ids).sum()
        out.backward()

        # Gradient for padding_idx should be zero
        self.assertTrue(torch.all(emb.weight.grad[padding_idx] == 0))


class TestConv2d(unittest.TestCase):
    """Tests for CPUBouncingConv2d."""

    @requires_cuda
    def test_forward_matches_nn_conv2d(self):
        """Test that forward pass matches nn.Conv2d."""
        from ramtorch.modules.conv import Conv2d

        in_channels, out_channels = 3, 16
        kernel_size = 3
        batch_size, height, width = 2, 32, 32

        torch.manual_seed(42)
        nn_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)

        ramtorch_conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            device="cuda",
            _weight=nn_conv.weight.data.clone(),
            _bias=nn_conv.bias.data.clone(),
        )
        nn_conv = nn_conv.cuda()

        x = torch.randn(batch_size, in_channels, height, width, device="cuda")

        nn_out = nn_conv(x)
        ramtorch_out = ramtorch_conv(x)

        self.assertTrue(torch.allclose(nn_out, ramtorch_out, atol=1e-4))

    @requires_cuda
    def test_backward_gradient_shapes(self):
        """Test that backward pass produces correct gradient shapes."""
        from ramtorch.modules.conv import Conv2d

        in_channels, out_channels = 3, 16
        kernel_size = 3

        conv = Conv2d(in_channels, out_channels, kernel_size, padding=1, device="cuda")
        x = torch.randn(2, in_channels, 16, 16, device="cuda", requires_grad=True)

        out = conv(x).sum()
        out.backward()

        self.assertEqual(conv.weight.grad.shape, conv.weight.shape)
        self.assertEqual(conv.bias.grad.shape, conv.bias.shape)

    @requires_cuda
    def test_no_bias(self):
        """Test Conv2d without bias."""
        from ramtorch.modules.conv import Conv2d

        conv = Conv2d(3, 16, 3, bias=False, device="cuda")
        x = torch.randn(2, 3, 16, 16, device="cuda")

        out = conv(x)
        self.assertEqual(out.shape, (2, 16, 14, 14))
        self.assertIsNone(conv.bias)


class TestConv3d(unittest.TestCase):
    """Tests for CPUBouncingConv3d."""

    @requires_cuda
    def test_forward_matches_nn_conv3d(self):
        """Test that forward pass matches nn.Conv3d."""
        from ramtorch.modules.conv import Conv3d

        in_channels, out_channels = 3, 8
        kernel_size = 3
        batch_size, depth, height, width = 2, 8, 16, 16

        torch.manual_seed(42)
        nn_conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1)

        ramtorch_conv = Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            device="cuda",
            _weight=nn_conv.weight.data.clone(),
            _bias=nn_conv.bias.data.clone(),
        )
        nn_conv = nn_conv.cuda()

        x = torch.randn(batch_size, in_channels, depth, height, width, device="cuda")

        nn_out = nn_conv(x)
        ramtorch_out = ramtorch_conv(x)

        self.assertTrue(torch.allclose(nn_out, ramtorch_out, atol=1e-4))


class TestLayerNorm(unittest.TestCase):
    """Tests for CPUBouncingLayerNorm."""

    @requires_cuda
    def test_forward_matches_nn_layernorm(self):
        """Test that forward pass matches nn.LayerNorm."""
        from ramtorch.modules.norm import LayerNorm

        normalized_shape = (64,)
        batch_size, seq_len = 4, 10

        torch.manual_seed(42)
        nn_ln = nn.LayerNorm(normalized_shape)

        ramtorch_ln = LayerNorm(
            normalized_shape,
            device="cuda",
            _weight=nn_ln.weight.data.clone(),
            _bias=nn_ln.bias.data.clone(),
        )
        nn_ln = nn_ln.cuda()

        x = torch.randn(batch_size, seq_len, 64, device="cuda")

        nn_out = nn_ln(x)
        ramtorch_out = ramtorch_ln(x)

        self.assertTrue(torch.allclose(nn_out, ramtorch_out, atol=1e-5))

    @requires_cuda
    def test_backward_gradient_shapes(self):
        """Test that backward pass produces correct gradient shapes."""
        from ramtorch.modules.norm import LayerNorm

        ln = LayerNorm((64,), device="cuda")
        x = torch.randn(4, 10, 64, device="cuda", requires_grad=True)

        out = ln(x).sum()
        out.backward()

        self.assertEqual(ln.weight.grad.shape, ln.weight.shape)
        self.assertEqual(ln.bias.grad.shape, ln.bias.shape)

    @requires_cuda
    def test_no_affine(self):
        """Test LayerNorm without elementwise_affine."""
        from ramtorch.modules.norm import LayerNorm

        ln = LayerNorm((64,), elementwise_affine=False, device="cuda")
        x = torch.randn(4, 10, 64, device="cuda")

        out = ln(x)
        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(ln.weight)
        self.assertIsNone(ln.bias)


class TestRMSNorm(unittest.TestCase):
    """Tests for CPUBouncingRMSNorm."""

    @requires_cuda
    def test_forward_computation(self):
        """Test that RMSNorm computes correctly."""
        from ramtorch.modules.norm import RMSNorm

        normalized_shape = (64,)
        batch_size, seq_len = 4, 10

        rms = RMSNorm(normalized_shape, device="cuda")
        x = torch.randn(batch_size, seq_len, 64, device="cuda")

        out = rms(x)

        # Manually compute RMSNorm
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + 1e-6)
        expected = (x_norm * rms.weight.to("cuda")).to(x.dtype)

        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(out, expected, atol=1e-4))

    @requires_cuda
    def test_weight_addition_mode(self):
        """Test Gemma-style weight addition mode."""
        from ramtorch.modules.norm import RMSNorm

        normalized_shape = (64,)

        # With weight addition, weight should be initialized to 0
        rms = RMSNorm(normalized_shape, use_weight_addition=True, device="cuda")

        self.assertTrue(torch.all(rms.weight == 0))

        x = torch.randn(4, 10, 64, device="cuda")
        out = rms(x)

        # With weight=0 and use_weight_addition, output should be x_norm * (1 + 0) = x_norm
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        expected = (x * torch.rsqrt(variance + 1e-6)).to(x.dtype)

        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(out, expected, atol=1e-4))

    @requires_cuda
    def test_backward_gradient_shapes(self):
        """Test that backward pass produces correct gradient shapes."""
        from ramtorch.modules.norm import RMSNorm

        rms = RMSNorm((64,), device="cuda")
        x = torch.randn(4, 10, 64, device="cuda", requires_grad=True)

        out = rms(x).sum()
        out.backward()

        self.assertEqual(rms.weight.grad.shape, rms.weight.shape)


class TestReplacementFunctions(unittest.TestCase):
    """Tests for model replacement helper functions."""

    @requires_cuda
    def test_replace_linear_with_ramtorch(self):
        """Test replacing nn.Linear with RamTorch Linear."""
        from ramtorch.helpers import replace_linear_with_ramtorch
        from ramtorch.modules.linear import Linear

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        replace_linear_with_ramtorch(model, device="cuda")

        # Check that Linear layers were replaced
        self.assertIsInstance(model[0], Linear)
        self.assertIsInstance(model[2], Linear)
        self.assertIsInstance(model[1], nn.ReLU)

    @requires_cuda
    def test_replace_all_with_ramtorch(self):
        """Test replacing all supported layers."""
        from ramtorch.helpers import replace_all_with_ramtorch
        from ramtorch.modules.linear import Linear
        from ramtorch.modules.embedding import Embedding
        from ramtorch.modules.norm import LayerNorm

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 128)
                self.norm = nn.LayerNorm(128)

            def forward(self, x):
                x = self.embed(x)
                x = self.linear(x)
                return self.norm(x)

        model = SimpleModel()
        counts = replace_all_with_ramtorch(model, device="cuda")

        self.assertIsInstance(model.embed, Embedding)
        self.assertIsInstance(model.linear, Linear)
        self.assertIsInstance(model.norm, LayerNorm)

        self.assertEqual(counts["linear"], 1)
        self.assertEqual(counts["embedding"], 1)
        self.assertEqual(counts["layernorm"], 1)


class TestIsRamtorchAttribute(unittest.TestCase):
    """Tests for the is_ramtorch attribute on parameters."""

    @requires_cuda
    def test_linear_has_is_ramtorch(self):
        """Test that Linear parameters have is_ramtorch attribute."""
        from ramtorch.modules.linear import Linear

        linear = Linear(64, 128, device="cuda")

        self.assertTrue(getattr(linear.weight, "is_ramtorch", False))
        self.assertTrue(getattr(linear.bias, "is_ramtorch", False))

    @requires_cuda
    def test_embedding_has_is_ramtorch(self):
        """Test that Embedding parameters have is_ramtorch attribute."""
        from ramtorch.modules.embedding import Embedding

        emb = Embedding(100, 64, device="cuda")

        self.assertTrue(getattr(emb.weight, "is_ramtorch", False))

    @requires_cuda
    def test_conv2d_has_is_ramtorch(self):
        """Test that Conv2d parameters have is_ramtorch attribute."""
        from ramtorch.modules.conv import Conv2d

        conv = Conv2d(3, 16, 3, device="cuda")

        self.assertTrue(getattr(conv.weight, "is_ramtorch", False))
        self.assertTrue(getattr(conv.bias, "is_ramtorch", False))


if __name__ == "__main__":
    unittest.main()
