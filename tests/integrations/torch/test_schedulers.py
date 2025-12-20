"""Tests for learning rate schedulers module."""

import math

import torch
import torch.nn as nn

from formed.integrations.torch.schedulers import CosineLRScheduler


class TestCosineLRScheduler:
    def test_basic_cosine_schedule(self):
        """Test basic cosine annealing without warmup or restarts."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=0.0,
        )

        # Check initial learning rate
        assert optimizer.param_groups[0]["lr"] == 1.0

        # Step through half cycle - should be around lr_min + (lr_max - lr_min) * 0.5
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        expected_lr = 0.5 * (1 + math.cos(math.pi * 5 / 10))
        assert abs(lr - expected_lr) < 1e-6

        # Step to end of cycle - should approach lr_min
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        assert lr < 0.01  # Should be close to lr_min

    def test_warmup_phase(self):
        """Test linear warmup phase."""
        model = nn.Linear(10, 1)
        base_lr = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

        warmup_t = 5
        warmup_lr_init = 0.01

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=0.0,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
        )

        # During warmup, learning rate should increase linearly
        # Note: optimizer.step() should be called before scheduler.step()
        # After step(), last_epoch increases from 0 to 1, 2, 3, ...
        # Warmup lasts while last_epoch < warmup_t, i.e., last_epoch = 1, 2, 3, 4
        # So we need warmup_t-1 scheduler steps to complete warmup
        lrs = []
        for step in range(warmup_t + 2):  # Go a bit past warmup
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # Warmup completes after warmup_t-1 steps (last_epoch reaches warmup_t-1)
        # Check that learning rate increases monotonically during warmup
        warmup_lrs = lrs[: warmup_t - 1]
        assert all(warmup_lrs[i] < warmup_lrs[i + 1] for i in range(len(warmup_lrs) - 1)), (
            f"Warmup LRs not monotonic: {warmup_lrs}"
        )

        # Check first warmup step (after first scheduler.step(), last_epoch=1)
        # At last_epoch=1: warmup_lr_init + (1+1) * step = warmup_lr_init + 2 * (base_lr - warmup_lr_init) / warmup_t
        expected_first_lr = warmup_lr_init + 2 * (base_lr - warmup_lr_init) / warmup_t
        assert abs(lrs[0] - expected_first_lr) < 1e-6

        # Check last warmup step (last_epoch=warmup_t-1) should reach base_lr
        # At last_epoch=warmup_t-1: warmup_lr_init + warmup_t * (base_lr - warmup_lr_init) / warmup_t = base_lr
        expected_last_warmup_lr = base_lr
        assert abs(lrs[warmup_t - 2] - expected_last_warmup_lr) < 1e-6

    def test_cycle_restarts(self):
        """Test learning rate restarts with cycle_mul."""
        model = nn.Linear(10, 1)
        base_lr = 1.0
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=5,
            lr_min=0.0,
            cycle_mul=2.0,  # Each cycle is 2x longer
            cycle_limit=2,
        )

        lrs = []
        for _ in range(20):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # First cycle: steps 0-4
        # Within first cycle, LR should vary (cosine pattern)
        assert lrs[0] < lrs[4]  # At step 4, approaching end of first cycle (high LR)

        # Second cycle starts at step 5
        # Within second cycle, should see cosine decay
        assert lrs[5] > lrs[14]  # Step 5 is near start, step 14 is near end of second cycle

        # Check that we have restarts (LR doesn't monotonically decrease)
        # Find local maxima
        has_restart = False
        for i in range(1, len(lrs) - 1):
            if lrs[i - 1] < lrs[i] and lrs[i] > lrs[i + 1]:
                has_restart = True
                break
        assert has_restart, "Should have at least one restart"

    def test_cycle_decay(self):
        """Test learning rate decay across cycles."""
        model = nn.Linear(10, 1)
        base_lr = 1.0
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

        cycle_decay = 0.5
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=5,
            lr_min=0.0,
            cycle_mul=1.0,
            cycle_decay=cycle_decay,
            cycle_limit=3,
        )

        # Get LR at start of each cycle
        cycle_start_lrs = []

        for step in range(16):
            lr = optimizer.param_groups[0]["lr"]
            if step % 5 == 0:  # Start of cycle
                cycle_start_lrs.append(lr)
            optimizer.step()
            scheduler.step()

        # Each cycle should start with base_lr * cycle_decay^cycle_index
        assert abs(cycle_start_lrs[0] - base_lr) < 1e-6
        assert abs(cycle_start_lrs[1] - base_lr * cycle_decay) < 1e-6
        assert abs(cycle_start_lrs[2] - base_lr * cycle_decay**2) < 1e-6

    def test_cycle_limit(self):
        """Test that scheduler respects cycle_limit."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        lr_min = 0.01
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=5,
            lr_min=lr_min,
            cycle_mul=1.0,
            cycle_limit=2,  # Only 2 cycles
        )

        # Run past cycle_limit
        for _ in range(20):
            optimizer.step()
            scheduler.step()

        # After cycle_limit, should stay at lr_min
        lr = optimizer.param_groups[0]["lr"]
        assert abs(lr - lr_min) < 1e-6

    def test_warmup_prefix(self):
        """Test warmup_prefix parameter."""
        model = nn.Linear(10, 1)
        base_lr = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)

        warmup_t = 3
        t_initial = 10

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=0.0,
            warmup_t=warmup_t,
            warmup_lr_init=0.01,
            warmup_prefix=True,  # Warmup doesn't count toward t_initial
        )

        # With warmup_prefix=True, cosine cycle starts after warmup
        # So at step warmup_t, we should be at the beginning of cosine cycle
        for _ in range(warmup_t):
            optimizer.step()
            scheduler.step()

        lr_after_warmup = optimizer.param_groups[0]["lr"]

        # Should be at base_lr (start of cosine cycle)
        assert abs(lr_after_warmup - base_lr) < 1e-6

    def test_multiple_param_groups(self):
        """Test scheduler with multiple parameter groups."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        optimizer = torch.optim.SGD(
            [
                {"params": model[0].parameters(), "lr": 0.1},
                {"params": model[1].parameters(), "lr": 0.01},
            ]
        )

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=0.0,
        )

        # Check initial learning rates
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[1]["lr"] == 0.01

        # Step and check that both groups are scheduled
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        lr1 = optimizer.param_groups[0]["lr"]
        lr2 = optimizer.param_groups[1]["lr"]

        # Both should have decreased but maintain ratio
        assert lr1 < 0.1
        assert lr2 < 0.01
        assert abs(lr1 / lr2 - 10.0) < 1e-5  # Ratio should be maintained

    def test_state_dict_and_load(self):
        """Test saving and loading scheduler state."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            lr_min=0.0,
            warmup_t=2,
            warmup_lr_init=0.01,
        )

        # Step a few times
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        last_epoch_before = scheduler.last_epoch
        state = scheduler.state_dict()

        # Create new scheduler and load state
        optimizer2 = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler2 = CosineLRScheduler(
            optimizer2,
            t_initial=10,
            lr_min=0.0,
            warmup_t=2,
            warmup_lr_init=0.01,
        )
        scheduler2.load_state_dict(state)

        # Check that last_epoch was restored
        assert scheduler2.last_epoch == last_epoch_before

        # Step once more and compare
        optimizer.step()
        scheduler.step()
        lr_next1 = optimizer.param_groups[0]["lr"]

        optimizer2.step()
        scheduler2.step()
        lr_next2 = optimizer2.param_groups[0]["lr"]

        # Both should produce the same next learning rate
        assert abs(lr_next1 - lr_next2) < 1e-6

    def test_get_cycle_length(self):
        """Test get_cycle_length method."""
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Test with cycle_mul=1.0
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=10,
            cycle_mul=1.0,
            cycle_limit=3,
        )

        # 3 cycles of 10 steps each
        assert scheduler.get_cycle_length(3) == 30

        # Test with cycle_mul=2.0
        scheduler2 = CosineLRScheduler(
            optimizer,
            t_initial=10,
            cycle_mul=2.0,
            cycle_limit=3,
        )

        # First cycle: 10, second: 20, third: 40
        # Total: 10 + 20 + 40 = 70
        expected = 10 + 20 + 40
        assert scheduler2.get_cycle_length(3) == expected

        # Test with warmup_prefix
        scheduler3 = CosineLRScheduler(
            optimizer,
            t_initial=10,
            cycle_mul=1.0,
            warmup_t=5,
            warmup_prefix=True,
        )

        # Should include warmup in total length
        assert scheduler3.get_cycle_length(2) == 20 + 5
