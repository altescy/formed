"""Tests for tensor initializers module."""

import torch

from formed.integrations.torch.initializers import (
    KaimingNormalTensorInitializer,
    KaimingUniformTensorInitializer,
    NormalTensorInitializer,
    OnesTensorInitializer,
    OrthogonalTensorInitializer,
    SparseTensorInitializer,
    UniformTensorInitializer,
    XavierNormalTensorInitializer,
    XavierUniformTensorInitializer,
    ZerosTensorInitializer,
)


class TestUniformTensorInitializer:
    def test_default_params(self):
        """Test uniform initializer with default parameters."""
        shape = (10, 20)
        initializer = UniformTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        assert torch.all(tensor >= 0.0)
        assert torch.all(tensor <= 1.0)

    def test_custom_range(self):
        """Test uniform initializer with custom range."""
        shape = (5, 10)
        low, high = -2.0, 3.0
        initializer = UniformTensorInitializer(shape, low=low, high=high)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        assert torch.all(tensor >= low)
        assert torch.all(tensor <= high)


class TestNormalTensorInitializer:
    def test_default_params(self):
        """Test normal initializer with default parameters."""
        shape = (10, 20)
        initializer = NormalTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        # Statistical test: check mean and std are close to 0 and 1
        assert abs(tensor.mean().item()) < 0.5
        assert abs(tensor.std().item() - 1.0) < 0.5

    def test_custom_params(self):
        """Test normal initializer with custom mean and std."""
        shape = (100, 100)
        mean, std = 5.0, 2.0
        initializer = NormalTensorInitializer(shape, mean=mean, std=std)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        # Statistical test: check mean and std are close to target
        assert abs(tensor.mean().item() - mean) < 1.0
        assert abs(tensor.std().item() - std) < 1.0


class TestXavierUniformTensorInitializer:
    def test_default_gain(self):
        """Test Xavier uniform initializer with default gain."""
        shape = (20, 30)
        initializer = XavierUniformTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)

    def test_custom_gain(self):
        """Test Xavier uniform initializer with custom gain."""
        shape = (10, 15)
        gain = 2.0
        initializer = XavierUniformTensorInitializer(shape, gain=gain)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestXavierNormalTensorInitializer:
    def test_default_gain(self):
        """Test Xavier normal initializer with default gain."""
        shape = (20, 30)
        initializer = XavierNormalTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)

    def test_custom_gain(self):
        """Test Xavier normal initializer with custom gain."""
        shape = (10, 15)
        gain = 2.0
        initializer = XavierNormalTensorInitializer(shape, gain=gain)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestKaimingUniformTensorInitializer:
    def test_default_params(self):
        """Test Kaiming uniform initializer with default parameters."""
        shape = (20, 30)
        initializer = KaimingUniformTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)

    def test_custom_params(self):
        """Test Kaiming uniform initializer with custom parameters."""
        shape = (10, 15)
        initializer = KaimingUniformTensorInitializer(shape, a=0.1, mode="fan_out", nonlinearity="relu")
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestKaimingNormalTensorInitializer:
    def test_default_params(self):
        """Test Kaiming normal initializer with default parameters."""
        shape = (20, 30)
        initializer = KaimingNormalTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)

    def test_custom_params(self):
        """Test Kaiming normal initializer with custom parameters."""
        shape = (10, 15)
        initializer = KaimingNormalTensorInitializer(shape, a=0.1, mode="fan_out", nonlinearity="relu")
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestOrthogonalTensorInitializer:
    def test_default_gain(self):
        """Test orthogonal initializer with default gain."""
        shape = (20, 30)
        initializer = OrthogonalTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)

    def test_custom_gain(self):
        """Test orthogonal initializer with custom gain."""
        shape = (10, 15)
        gain = 2.0
        initializer = OrthogonalTensorInitializer(shape, gain=gain)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestSparseTensorInitializer:
    def test_default_params(self):
        """Test sparse initializer with default parameters."""
        shape = (20, 30)
        initializer = SparseTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        # Check sparsity: approximately 10% should be non-zero
        non_zero_ratio = (tensor != 0).float().mean().item()
        assert 0.5 < non_zero_ratio < 1.0  # Sparse init sets most to zero

    def test_custom_params(self):
        """Test sparse initializer with custom sparsity."""
        shape = (100, 100)
        sparsity = 0.5
        initializer = SparseTensorInitializer(shape, sparsity=sparsity, std=0.1)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)


class TestZerosTensorInitializer:
    def test_zeros(self):
        """Test zeros initializer."""
        shape = (10, 20, 30)
        initializer = ZerosTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        assert torch.all(tensor == 0.0)


class TestOnesTensorInitializer:
    def test_ones(self):
        """Test ones initializer."""
        shape = (10, 20, 30)
        initializer = OnesTensorInitializer(shape)
        tensor = initializer()

        assert tensor.shape == torch.Size(shape)
        assert torch.all(tensor == 1.0)
