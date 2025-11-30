"""Tests for formed.common.iterutils module"""

import multiprocessing
import time
from collections.abc import Iterable, Iterator

import pytest

from formed.common.iterutils import BufferedIterator, SizedIterator, batched


class ErrorIterator:
    """Picklable iterator that raises an error after 5 items"""

    def __init__(self) -> None:
        self.count = 0

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self.count >= 5:
            raise ValueError("Test error")
        value = self.count
        self.count += 1
        return value


class SlowIterator:
    """Picklable iterator that simulates slow iteration"""

    def __init__(self, n: int) -> None:
        self.n = n
        self.current = 0

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        if self.current >= self.n:
            raise StopIteration
        value = self.current
        self.current += 1
        time.sleep(0.01)
        return value


class TestSizedIterator:
    """Tests for SizedIterator class"""

    def test_sized_iterator_basic(self) -> None:
        """Test basic functionality of SizedIterator"""
        data = list(range(10))
        iterator = SizedIterator(iter(data), len(data))

        assert len(iterator) == 10
        result = list(iterator)
        assert result == data

    def test_sized_iterator_size(self) -> None:
        """Test that size is preserved correctly"""
        data = list(range(100))
        iterator = SizedIterator(iter(data), len(data))

        assert iterator.size == 100
        assert len(iterator) == 100

    def test_sized_iterator_iter(self) -> None:
        """Test __iter__ returns the inner iterator"""
        data = list(range(5))
        inner_iter = iter(data)
        sized_iter = SizedIterator(inner_iter, len(data))

        assert iter(sized_iter) is inner_iter

    def test_sized_iterator_next(self) -> None:
        """Test __next__ method"""
        data = list(range(5))
        iterator = SizedIterator(iter(data), len(data))

        assert next(iterator) == 0
        assert next(iterator) == 1
        assert next(iterator) == 2

    def test_sized_iterator_exhaustion(self) -> None:
        """Test that iterator raises StopIteration when exhausted"""
        data = list(range(3))
        iterator = SizedIterator(iter(data), len(data))

        list(iterator)  # Exhaust the iterator

        with pytest.raises(StopIteration):
            next(iterator)


class TestBufferedIterator:
    """Tests for BufferedIterator class"""

    def test_buffered_iterator_simple(self) -> None:
        """Test BufferedIterator with simple data"""
        data = list(range(20))
        buffered = BufferedIterator(iter(data), buffer_size=5)

        result = list(buffered)
        assert result == data

    def test_buffered_iterator_lazy_initialization(self) -> None:
        """Test that process starts lazily on first next() call"""
        data = list(range(10))
        buffered = BufferedIterator(iter(data), buffer_size=5)

        # Process should not be started yet
        assert buffered._process is None

        # First next() should start the process
        first_item = next(buffered)
        assert first_item == 0
        assert buffered._process is not None
        assert buffered._process.is_alive()

        # Clean up
        buffered.close()

    def test_buffered_iterator_with_sized_iterator_wrapper(self) -> None:
        """Test BufferedIterator wrapped in SizedIterator"""
        data = list(range(15))
        buffered = BufferedIterator(iter(data), buffer_size=5)
        sized = SizedIterator(buffered, len(data))

        # Process should start when iterating
        result = list(sized)
        assert result == data

    def test_buffered_iterator_closes_properly(self) -> None:
        """Test that BufferedIterator closes the process properly"""
        data = list(range(100))
        buffered = BufferedIterator(iter(data), buffer_size=10)

        # Start iteration
        next(buffered)
        assert buffered._process is not None
        process = buffered._process

        # Close
        buffered.close()

        # Process should be terminated
        time.sleep(0.2)  # Give it time to terminate
        assert not process.is_alive()

    def test_buffered_iterator_partial_iteration(self) -> None:
        """Test that BufferedIterator handles partial iteration correctly"""
        data = list(range(50))
        buffered = BufferedIterator(iter(data), buffer_size=10)

        # Read only first 10 items
        result = []
        for i, item in enumerate(buffered):
            result.append(item)
            if i >= 9:
                break

        assert result == list(range(10))

        # Clean up
        buffered.close()

    def test_buffered_iterator_exception_handling(self) -> None:
        """Test that BufferedIterator handles exceptions from source iterator

        Note: Uses a picklable iterator class defined at module level.
        """
        buffered = BufferedIterator(ErrorIterator(), buffer_size=3)

        # Should get first 5 items
        result = []
        with pytest.raises(ValueError, match="Test error"):
            for item in buffered:
                result.append(item)

        assert result == [0, 1, 2, 3, 4]

    def test_buffered_iterator_empty_source(self) -> None:
        """Test BufferedIterator with empty source"""
        buffered = BufferedIterator(iter([]), buffer_size=5)

        result = list(buffered)
        assert result == []

    def test_buffered_iterator_context_manager_usage(self) -> None:
        """Test using BufferedIterator with context manager pattern

        Note: Uses a picklable iterator class defined at module level.
        """
        buffered = BufferedIterator(SlowIterator(10), buffer_size=5)

        try:
            result = []
            for i, item in enumerate(buffered):
                result.append(item)
                if i >= 4:
                    break

            assert result == [0, 1, 2, 3, 4]
        finally:
            buffered.close()


class TestBatched:
    """Tests for batched function"""

    def test_batched_basic(self) -> None:
        """Test basic batching functionality"""
        data = list(range(10))
        batches = list(batched(data, batch_size=3))

        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_batched_exact_division(self) -> None:
        """Test batching when data divides evenly"""
        data = list(range(12))
        batches = list(batched(data, batch_size=4))

        assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def test_batched_drop_last(self) -> None:
        """Test batching with drop_last=True"""
        data = list(range(10))
        batches = list(batched(data, batch_size=3, drop_last=True))

        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_batched_single_batch(self) -> None:
        """Test batching when batch_size >= data length"""
        data = list(range(5))
        batches = list(batched(data, batch_size=10))

        assert batches == [[0, 1, 2, 3, 4]]

    def test_batched_batch_size_one(self) -> None:
        """Test batching with batch_size=1"""
        data = list(range(3))
        batches = list(batched(data, batch_size=1))

        assert batches == [[0], [1], [2]]

    def test_batched_empty_iterable(self) -> None:
        """Test batching empty iterable"""
        batches = list(batched([], batch_size=5))

        assert batches == []

    def test_batched_sized_iterator(self) -> None:
        """Test that batched returns SizedIterator for sized iterables"""
        data = list(range(10))
        result = batched(data, batch_size=3)

        assert isinstance(result, SizedIterator)
        assert len(result) == 4  # ceil(10/3) = 4

    def test_batched_generator(self) -> None:
        """Test batching with generator (non-sized iterable)"""

        def gen() -> Iterable[int]:
            for i in range(10):
                yield i

        batches = list(batched(gen(), batch_size=3))

        assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


if __name__ == "__main__":
    # For debugging: force spawn method for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    pytest.main([__file__, "-v"])
