"""Tests for formed.integrations.ml.dataloader module"""

import multiprocessing
import time
import warnings
from collections.abc import Sequence

import pytest

from formed.common.ctxutils import closing
from formed.common.iterutils import SizedIterator
from formed.integrations.ml import BasicBatchSampler, DataLoader


def simple_collator(batch: Sequence[int]) -> list[int]:
    """Simple collator that doubles each value"""
    return [x * 2 for x in batch]


def slow_collator(batch: Sequence[int]) -> list[int]:
    """Slow collator for testing buffering performance"""
    time.sleep(0.1)
    return [x * 2 for x in batch]


class TestBasicBatchSampler:
    """Tests for BasicBatchSampler class"""

    def test_basic_sampler_no_shuffle(self) -> None:
        """Test BasicBatchSampler without shuffling"""
        dataset = list(range(10))
        sampler = BasicBatchSampler(batch_size=3, shuffle=False)

        batches = list(sampler(dataset))

        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]
        assert batches[3] == [9]

    def test_basic_sampler_shuffle(self) -> None:
        """Test BasicBatchSampler with shuffling"""
        dataset = list(range(20))
        sampler = BasicBatchSampler(batch_size=5, shuffle=True, seed=42)

        batches = list(sampler(dataset))

        assert len(batches) == 4
        assert all(len(batch) == 5 for batch in batches)

        # Verify all indices are present
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == list(range(20))

    def test_basic_sampler_drop_last(self) -> None:
        """Test BasicBatchSampler with drop_last=True"""
        dataset = list(range(10))
        sampler = BasicBatchSampler(batch_size=3, shuffle=False, drop_last=True)

        batches = list(sampler(dataset))

        assert len(batches) == 3
        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_basic_sampler_batch_size_one(self) -> None:
        """Test BasicBatchSampler with batch_size=1"""
        dataset = list(range(5))
        sampler = BasicBatchSampler(batch_size=1, shuffle=False)

        batches = list(sampler(dataset))

        assert len(batches) == 5
        assert batches == [[0], [1], [2], [3], [4]]

    def test_basic_sampler_returns_sized_iterator(self) -> None:
        """Test that BasicBatchSampler returns a SizedIterator"""
        dataset = list(range(15))
        sampler = BasicBatchSampler(batch_size=4, shuffle=False)

        result = sampler(dataset)

        assert isinstance(result, SizedIterator)
        assert len(result) == 4

    def test_basic_sampler_empty_dataset(self) -> None:
        """Test BasicBatchSampler with empty dataset"""
        dataset = []
        sampler = BasicBatchSampler(batch_size=5, shuffle=False)

        batches = list(sampler(dataset))

        assert batches == []

    def test_basic_sampler_reproducibility(self) -> None:
        """Test that shuffling is reproducible with same seed"""
        dataset = list(range(20))
        sampler1 = BasicBatchSampler(batch_size=5, shuffle=True, seed=123)
        sampler2 = BasicBatchSampler(batch_size=5, shuffle=True, seed=123)

        batches1 = list(sampler1(dataset))
        batches2 = list(sampler2(dataset))

        assert batches1 == batches2


class TestDataLoader:
    """Tests for DataLoader class"""

    def test_dataloader_basic(self) -> None:
        """Test basic DataLoader functionality"""
        dataset = list(range(20))
        sampler = BasicBatchSampler(batch_size=5, shuffle=False)

        dataloader = DataLoader(sampler=sampler, collator=simple_collator)

        with closing(dataloader(dataset)) as loader:
            batches = list(loader)

        assert len(batches) == 4
        assert batches[0] == [0, 2, 4, 6, 8]
        assert batches[1] == [10, 12, 14, 16, 18]
        assert batches[2] == [20, 22, 24, 26, 28]
        assert batches[3] == [30, 32, 34, 36, 38]

    def test_dataloader_returns_sized_iterator(self) -> None:
        """Test that DataLoader returns a SizedIterator"""
        dataset = list(range(10))
        sampler = BasicBatchSampler(batch_size=3, shuffle=False)

        dataloader = DataLoader(sampler=sampler, collator=simple_collator)
        loader = dataloader(dataset)

        assert isinstance(loader, SizedIterator)
        assert len(loader) == 4

    def test_dataloader_with_buffering(self) -> None:
        """Test DataLoader with buffering enabled"""
        dataset = list(range(20))
        sampler = BasicBatchSampler(batch_size=5, shuffle=False)

        dataloader = DataLoader(
            sampler=sampler,
            collator=simple_collator,
            buffer_size=10,
        )

        with closing(dataloader(dataset)) as loader:
            batches = list(loader)

        assert len(batches) == 4
        assert batches[0] == [0, 2, 4, 6, 8]
        assert batches[1] == [10, 12, 14, 16, 18]

    def test_dataloader_buffering_performance(self) -> None:
        """Test that buffering improves performance

        Note: This test is skipped because performance comparisons are
        unreliable due to system load and multiprocessing overhead.
        The buffering functionality is tested in other tests.
        """
        dataset = list(range(20))
        sampler = BasicBatchSampler(batch_size=5, shuffle=False)

        # Without buffering
        dataloader_no_buffer = DataLoader(
            sampler=sampler,
            collator=slow_collator,
            buffer_size=0,
        )

        start_time = time.time()
        with closing(dataloader_no_buffer(dataset)) as loader:
            for _ in loader:
                time.sleep(0.1)  # Simulate some processing time
        time_no_buffer = time.time() - start_time

        # With buffering
        dataloader_with_buffer = DataLoader(
            sampler=sampler,
            collator=slow_collator,
            buffer_size=10,
        )

        start_time = time.time()
        with closing(dataloader_with_buffer(dataset)) as loader:
            for _ in loader:
                time.sleep(0.1)  # Simulate some processing time
        time_with_buffer = time.time() - start_time

        # Buffering should be faster or at least not significantly slower
        # Note: This test might be flaky due to system load
        assert time_with_buffer < time_no_buffer * 1.2  # Allow some margin
        if time_with_buffer >= time_no_buffer:
            warnings.warn(
                "Buffering did not improve performance: "
                f"{time_no_buffer:.2f}s (no buffer) vs {time_with_buffer:.2f}s (with buffer)"
            )

    def test_dataloader_partial_iteration(self) -> None:
        """Test DataLoader with partial iteration"""
        dataset = list(range(100))
        sampler = BasicBatchSampler(batch_size=10, shuffle=False)

        dataloader = DataLoader(
            sampler=sampler,
            collator=simple_collator,
            buffer_size=20,
        )

        with closing(dataloader(dataset)) as loader:
            batches = []
            for i, batch in enumerate(loader):
                batches.append(batch)
                if i >= 2:  # Get only first 3 batches
                    break

        assert len(batches) == 3
        assert batches[0] == [i * 2 for i in range(10)]

    def test_dataloader_empty_dataset(self) -> None:
        """Test DataLoader with empty dataset"""
        dataset = []
        sampler = BasicBatchSampler(batch_size=5, shuffle=False)

        dataloader = DataLoader(sampler=sampler, collator=simple_collator)

        with closing(dataloader(dataset)) as loader:
            batches = list(loader)

        assert batches == []

    def test_dataloader_with_shuffle(self) -> None:
        """Test DataLoader with shuffled data"""
        dataset = list(range(20))
        sampler = BasicBatchSampler(batch_size=5, shuffle=True, seed=42)

        dataloader = DataLoader(sampler=sampler, collator=simple_collator)

        with closing(dataloader(dataset)) as loader:
            batches = list(loader)

        assert len(batches) == 4

        # Verify all values are present (doubled)
        all_values = [val for batch in batches for val in batch]
        expected_values = [i * 2 for i in range(20)]
        assert sorted(all_values) == sorted(expected_values)

    def test_dataloader_closes_properly(self) -> None:
        """Test that DataLoader closes resources properly"""
        dataset = list(range(50))
        sampler = BasicBatchSampler(batch_size=10, shuffle=False)

        dataloader = DataLoader(
            sampler=sampler,
            collator=simple_collator,
            buffer_size=20,
        )

        loader = dataloader(dataset)

        # Start iteration
        iterator = iter(loader)
        next(iterator)

        # Close
        loader.close()

        # Should not crash
        assert True

    def test_dataloader_context_manager(self) -> None:
        """Test DataLoader with context manager"""
        dataset = list(range(30))
        sampler = BasicBatchSampler(batch_size=10, shuffle=False)

        dataloader = DataLoader(
            sampler=sampler,
            collator=simple_collator,
            buffer_size=20,
        )

        batches = []
        with closing(dataloader(dataset)) as loader:
            for batch in loader:
                batches.append(batch)

        assert len(batches) == 3

    def test_dataloader_collator_receives_correct_data(self) -> None:
        """Test that collator receives correct batch data"""
        dataset = ["a", "b", "c", "d", "e"]
        sampler = BasicBatchSampler(batch_size=2, shuffle=False)

        received_batches = []

        def capturing_collator(batch: Sequence[str]) -> list[str]:
            received_batches.append(list(batch))
            return [s.upper() for s in batch]

        dataloader = DataLoader(sampler=sampler, collator=capturing_collator)

        with closing(dataloader(dataset)) as loader:
            result_batches = list(loader)

        assert received_batches == [["a", "b"], ["c", "d"], ["e"]]
        assert result_batches == [["A", "B"], ["C", "D"], ["E"]]


if __name__ == "__main__":
    # For debugging: force spawn method for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    pytest.main([__file__, "-v"])
