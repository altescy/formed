import os

import pytest
import torch
import torch.nn as nn

from formed.integrations.torch import (
    DataParallelDistributor,
    DistributedDataParallelDistributor,
    SingleDeviceDistributor,
)


class TestSingleDeviceDistributor:
    def test_single_device_cpu(self):
        distributor = SingleDeviceDistributor(device="cpu")
        assert distributor.device == torch.device("cpu")

        # Test reduce
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(distributor.reduce(tensor, "mean"), tensor)
        assert torch.allclose(distributor.reduce(tensor, "sum"), tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_device_cuda(self):
        distributor = SingleDeviceDistributor(device="cuda:0")
        assert distributor.device.type == "cuda"

        # Test model wrapping (should be no-op)
        model = nn.Linear(10, 5)
        wrapped = distributor.wrap_model(model)
        assert isinstance(wrapped, nn.Linear)


class TestDataParallelDistributor:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_data_parallel(self):
        distributor = DataParallelDistributor(device_ids=[0, 1])
        assert distributor.device.type == "cuda"

        # Test model wrapping
        model = nn.Linear(10, 5)
        wrapped = distributor.wrap_model(model)
        assert isinstance(wrapped, nn.DataParallel)


class TestDistributedDataParallelDistributor:
    def test_ddp_initialization_single_process(self):
        """Test DDP initialization in a single process (gloo backend for CPU)."""
        # Set environment variables for single process
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        distributor = None
        try:
            distributor = DistributedDataParallelDistributor(
                backend="gloo",  # Use gloo for CPU
                init_method="env://",
            )

            assert distributor.rank == 0
            assert distributor.world_size == 1
            assert distributor.local_rank == 0
            assert distributor.device.type == "cpu"

            # Test model wrapping
            model = nn.Linear(10, 5)
            wrapped = distributor.wrap_model(model)
            assert isinstance(wrapped, nn.parallel.DistributedDataParallel)

            # Test reduce operations
            tensor = torch.tensor([1.0, 2.0, 3.0])
            reduced = distributor.reduce(tensor.clone(), "sum")
            assert torch.allclose(reduced, tensor)

            reduced = distributor.reduce(tensor.clone(), "mean")
            assert torch.allclose(reduced, tensor)

            # Test barrier
            distributor.barrier()

        finally:
            # Cleanup
            if distributor is not None:
                distributor.cleanup()
            # Clear environment variables
            for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                os.environ.pop(key, None)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_with_cuda(self):
        """Test DDP initialization with CUDA."""
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"

        distributor = None
        try:
            distributor = DistributedDataParallelDistributor(
                backend="nccl",
                init_method="env://",
            )

            assert distributor.rank == 0
            assert distributor.world_size == 1
            assert distributor.local_rank == 0
            assert distributor.device.type == "cuda"

            # Test model wrapping
            model = nn.Linear(10, 5).to(distributor.device)
            wrapped = distributor.wrap_model(model)
            assert isinstance(wrapped, nn.parallel.DistributedDataParallel)

        finally:
            # Cleanup
            if distributor is not None:
                distributor.cleanup()
            # Clear environment variables
            for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                os.environ.pop(key, None)
