"""Memory-efficient, disk-backed dataset with multiprocess support.

This module provides the Dataset class for storing and accessing large collections
of serialized objects that don't fit in memory.

Available Classes:
    - Dataset: Disk-backed sequence with automatic cleanup and pickle support
    - DatasetMetadata: SQLite-based metadata manager with reference counting
    - Index: Binary index entry for fast object lookups

Features:
    - Store millions of items without loading everything into RAM
    - Share datasets across processes via pickle/IPC
    - Automatic cleanup of temporary datasets when all references are gone
    - O(1) random access via memory-mapped indices
    - SQLite-based reference counting for multiprocess safety
    - Paginated storage for efficient large-scale data handling

Performance:
    Dataset trades speed for memory efficiency:
    - Slower than in-memory list due to serialization and disk I/O overhead
    - Performance depends on object size and system I/O capabilities
    - Suitable for ML training where data loading isn't the bottleneck
    - Enables working with datasets larger than available RAM

    When to use Dataset:
    - Data doesn't fit in memory (e.g., millions of training examples)
    - Need to share data across multiple processes
    - Want automatic cleanup of temporary data

    When to use list:
    - Data fits comfortably in memory
    - Need maximum read/write speed
    - Single-process access only

Example:
    >>> from formed.common.dataset import Dataset
    >>>
    >>> # Create temporary dataset (auto-deleted when all references gone)
    >>> dataset = Dataset[dict]()
    >>> for i in range(1000000):
    ...     dataset.append({"id": i, "text": f"example_{i}"})
    >>> dataset.flush()
    >>> print(dataset[0])  # O(1) random access
    >>> print(len(dataset))  # 1000000
    >>>
    >>> # Persistent dataset
    >>> dataset = Dataset[dict](path="/path/to/dataset")
    >>> dataset.append({"data": "value"})
    >>> dataset.flush()
    >>> dataset.close()
    >>>
    >>> # Load existing dataset
    >>> dataset = Dataset.from_path("/path/to/dataset")
    >>>
    >>> # Multiprocess usage
    >>> import pickle
    >>> pickled = pickle.dumps(dataset)
    >>> # Send to child processes - each gets its own reference
    >>> # Dataset deleted only after all processes finish

"""

import shutil
import sqlite3
import tempfile
import uuid
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import cloudpickle as pickle

T = TypeVar("T")
Self = TypeVar("Self", bound="Dataset")


class DatasetMetadata:
    """SQLite-based metadata manager with reference counting for multiprocess safety.

    This class manages dataset metadata and tracks references to enable safe multiprocess
    access and automatic cleanup of temporary datasets. It uses SQLite with WAL mode for
    concurrent access from multiple processes.

    The metadata database contains:
    - Configuration metadata (e.g., pagesize)
    - Reference count entries tracking active Dataset instances

    When all references to a temporary dataset are removed, the dataset can be safely deleted.

    Args:
        db_path: Path to the SQLite database file (typically metadata.db in dataset directory).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection with proper settings for concurrent access."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                timeout=30.0,  # Wait up to 30 seconds for locks
                isolation_level="IMMEDIATE",  # Acquire write lock immediately
            )
            self._conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
            self._conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        return self._conn

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_references (
                    instance_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get_pagesize(self, default: int = 1024 * 1024 * 1024) -> int:
        """Get pagesize from metadata."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM metadata WHERE key = 'pagesize'")
        row = cursor.fetchone()
        if row is None:
            return default
        return int(row[0])

    def set_pagesize(self, pagesize: int) -> None:
        """Set pagesize in metadata."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('pagesize', ?)",
                (str(pagesize),),
            )

    def add_reference(self, instance_id: str) -> None:
        """Add a reference count entry."""
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO dataset_references (instance_id) VALUES (?)",
                (instance_id,),
            )

    def remove_reference(self, instance_id: str) -> None:
        """Remove a reference count entry."""
        conn = self._get_connection()
        with conn:
            conn.execute("DELETE FROM dataset_references WHERE instance_id = ?", (instance_id,))

    def get_reference_count(self) -> int:
        """Get total number of references."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM dataset_references")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class Index(NamedTuple):
    """Index entry for locating an object in the dataset's page files.

    Each index entry stores the location of a serialized object:
    - page: Which page file contains the object
    - offset: Byte offset within that page file
    - length: Size of the serialized object in bytes

    Indices are stored in a binary format (index.bin) for fast loading into memory.
    """

    page: int
    offset: int
    length: int

    def to_bytes(self) -> bytes:
        """Serialize index to 12-byte binary format (3 x 4-byte integers)."""
        return self.page.to_bytes(4, "little") + self.offset.to_bytes(4, "little") + self.length.to_bytes(4, "little")

    @classmethod
    def from_binaryio(cls, f: BinaryIO) -> "Index":
        """Deserialize index from binary file.

        Args:
            f: Binary file handle positioned at the start of an index entry.

        Returns:
            Index object reconstructed from the binary data.
        """
        return cls(
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
            int.from_bytes(f.read(4), "little"),
        )


class Dataset(Sequence[T]):
    """A memory-efficient, disk-backed dataset for storing and accessing large collections.

    Dataset provides efficient storage and retrieval of serialized objects with support for:
    - Large collections that don't fit in memory
    - Multiprocess access via SQLite-based reference counting
    - Pickle compatibility for IPC (inter-process communication)
    - Automatic cleanup of temporary datasets when all references are gone
    - Fast random access via memory-mapped indices

    The dataset stores data in paginated binary files and maintains an index for O(1) lookups.
    Metadata and reference counting are managed via SQLite for multiprocess safety.

    Examples:
        Create a temporary dataset (auto-deleted when all references are gone):
            >>> dataset = Dataset[str]()
            >>> dataset.append("hello")
            >>> dataset.append("world")
            >>> dataset.flush()
            >>> print(dataset[0])
            'hello'

        Create a persistent dataset:
            >>> dataset = Dataset[int](path="/path/to/dataset")
            >>> for i in range(1000):
            ...     dataset.append(i)
            >>> dataset.flush()

        Load an existing dataset:
            >>> dataset = Dataset.from_path("/path/to/dataset")
            >>> len(dataset)
            1000

        Use with multiprocessing (dataset is pickle-compatible):
            >>> import pickle
            >>> dataset = Dataset[str]()
            >>> dataset.append("shared")
            >>> pickled = pickle.dumps(dataset)
            >>> # Send pickled to another process...
            >>> restored = pickle.loads(pickled)
            >>> restored[0]
            'shared'

    Args:
        path: Optional path to store the dataset. If None, creates a temporary
            dataset that is automatically deleted when all references are gone.
        pagesize: Size of each page file in bytes. Default is 1GB.
            Larger pages reduce file fragmentation but may use more memory.
    """

    def __init__(
        self,
        path: Optional[Union[str, PathLike]] = None,
        pagesize: int = 1024 * 1024 * 1024,
    ) -> None:
        self._is_temporary = path is None
        if self._is_temporary:
            # Create temporary directory with a proper context manager
            self._tempdir = tempfile.mkdtemp(prefix="dataset_")
            self._path = Path(self._tempdir)
        else:
            assert path is not None
            self._tempdir = None
            self._path = Path(path)

        self._instance_id = str(uuid.uuid4())
        self._pagesize = pagesize
        self._indices: List[Index] = []
        self._pageios: Dict[int, BinaryIO] = {}
        self._metadata: Optional[DatasetMetadata] = None

        self._path.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _restore(self) -> None:
        # Initialize SQLite metadata
        db_filename = self._get_metadata_filename()
        self._metadata = DatasetMetadata(db_filename)

        # Load or set pagesize
        existing_pagesize = self._metadata.get_pagesize(default=self._pagesize)
        if existing_pagesize != self._pagesize and db_filename.stat().st_size > 0:
            # Database already exists, use its pagesize
            self._pagesize = existing_pagesize
        else:
            # New database or matching pagesize, store it
            self._metadata.set_pagesize(self._pagesize)

        # Add reference for this instance
        self._metadata.add_reference(self._instance_id)

        # Initialize index file
        index_filename = self._get_index_filename()
        if not index_filename.exists():
            index_filename.touch()

        self._indexio: BinaryIO = index_filename.open("rb+")
        if self._indexio.seek(0, 2) > 0:
            self._load_indices()

        # Open existing page files
        for page, page_filename in self._iter_page_filenames():
            self._pageios[page] = page_filename.open("rb+")

    def _cleanup(self) -> None:
        """Cleanup resources and remove reference count."""
        if not hasattr(self, "_metadata") or self._metadata is None:
            return

        try:
            # Close all file handles first
            if hasattr(self, "_pageios"):
                for pageio in self._pageios.values():
                    try:
                        pageio.close()
                    except Exception:
                        pass
                self._pageios.clear()

            if hasattr(self, "_indexio"):
                try:
                    self._indexio.close()
                except Exception:
                    pass

            # Remove this instance's reference
            self._metadata.remove_reference(self._instance_id)

            # Check if this was the last reference
            ref_count = self._metadata.get_reference_count()

            # Close metadata connection
            self._metadata.close()

            # If temporary and no more references, delete directory
            if self._is_temporary and ref_count == 0 and self._path.exists():
                shutil.rmtree(self._path)
        except Exception:
            # Suppress errors during cleanup to avoid issues at interpreter shutdown
            pass

    def __del__(self) -> None:
        """Cleanup when instance is garbage collected."""
        self._cleanup()

    @staticmethod
    def _encode(obj: T) -> bytes:
        return pickle.dumps(obj)

    @staticmethod
    def _decode(data: bytes) -> T:
        return cast(T, pickle.loads(data))

    @property
    def path(self) -> Path:
        """Get the filesystem path where the dataset is stored.

        Returns:
            Path object pointing to the dataset directory.
        """
        return self._path

    def _get_index_filename(self) -> Path:
        return self._path / "index.bin"

    def _get_metadata_filename(self) -> Path:
        return self._path / "metadata.db"

    def _get_page_filename(self, page: int) -> Path:
        return self._path / f"page_{page:08d}"

    def _iter_page_filenames(self) -> Iterable[Tuple[int, Path]]:
        for page_filename in self._path.glob("page_*"):
            page = int(page_filename.stem.split("_", 1)[1])
            yield page, page_filename

    def _add_index(self, index: Index) -> None:
        self._indices.append(index)
        self._indexio.seek(0, 2)
        self._indexio.write(index.to_bytes())

    def _load_indices(self) -> None:
        if self._indices:
            raise RuntimeError("indices already loaded")
        eof = self._indexio.seek(0, 2)
        self._indexio.seek(0)
        while self._indexio.tell() < eof:
            self._indices.append(Index.from_binaryio(self._indexio))

    def append(self, obj: T) -> None:
        """Append an object to the dataset.

        The object is serialized using cloudpickle and written to the current page.
        If the page would exceed the pagesize limit, a new page is created.

        Args:
            obj: The object to append. Must be serializable by cloudpickle.

        Note:
            Call flush() to ensure data is persisted to disk.
        """
        binary = self._encode(obj)

        pageio: BinaryIO
        if not self._pageios:
            page = 0
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio
        else:
            page = len(self._pageios) - 1
            pageio = self._pageios[page]

        offset = pageio.seek(0, 2)
        if offset + len(binary) > self._pagesize:
            page += 1
            offset = 0
            pageio = self._get_page_filename(page).open("wb+")
            self._pageios[page] = pageio

        pageio.write(binary)
        self._add_index(Index(page, offset, len(binary)))

    def flush(self) -> None:
        """Flush all buffered data to disk.

        This ensures that all appended objects and index entries are written
        to persistent storage. Call this periodically when appending large
        amounts of data, or before pickling the dataset.
        """
        for pageio in self._pageios.values():
            pageio.flush()
        self._indexio.flush()

    def close(self) -> None:
        """Close all open file handles but keep the dataset available.

        This closes file handles and database connections without removing
        the reference count. The dataset can be reopened by creating a new
        Dataset instance with the same path.

        Use this when you're done with a dataset but want to keep it on disk.
        For temporary datasets, all references must be deleted for cleanup.
        """
        for pageio in self._pageios.values():
            pageio.close()
        self._indexio.close()
        if self._metadata is not None:
            self._metadata.close()

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> List[T]: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[T, List[T]]:  # type: ignore[override]
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            index = self._indices[key]
            pageio = self._pageios[index.page]
            pageio.seek(index.offset)
            return self._decode(pageio.read(index.length))
        else:
            raise TypeError(f"key must be int or slice, not {type(key)}")

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare dataset for pickling. Supports temporary datasets."""
        # Flush before pickling but don't close (we still need this instance)
        self.flush()
        return {
            "path": str(self._path),
            "is_temporary": self._is_temporary,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore dataset from pickle. Creates new reference count."""
        self._path = Path(state["path"])
        self._is_temporary = state["is_temporary"]
        self._tempdir = None  # Don't track tempdir after unpickling
        self._instance_id = str(uuid.uuid4())  # New instance ID for new reference
        self._pagesize = 1024 * 1024 * 1024  # Will be overridden by metadata
        self._indices = []
        self._pageios = {}
        self._metadata = None
        self._restore()

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[T],
        path: Optional[Union[str, PathLike]] = None,
        pagesize: int = 1024 * 1024 * 1024,
    ) -> "Dataset[T]":
        """Create a dataset from an iterable.

        Convenience method that creates a dataset and appends all items
        from the iterable, then flushes the data to disk.

        Args:
            iterable: An iterable of objects to store in the dataset.
            path: Optional path to store the dataset. If None, creates a
                temporary dataset.
            pagesize: Size of each page file in bytes. Default is 1GB.

        Returns:
            A new Dataset containing all items from the iterable.

        Example:
            >>> data = [1, 2, 3, 4, 5]
            >>> dataset = Dataset.from_iterable(data, path="/tmp/numbers")
            >>> len(dataset)
            5
        """
        dataset = cls(path, pagesize)
        for obj in iterable:
            dataset.append(obj)
        dataset.flush()
        return dataset

    @classmethod
    def from_path(
        cls: Type[Self],
        path: Union[str, PathLike],
    ) -> Self:
        """Load an existing dataset from a path.

        This is equivalent to `Dataset(path=path)` but makes the intent
        clearer when loading existing datasets.

        Args:
            path: Path to an existing dataset directory.

        Returns:
            A Dataset instance pointing to the existing data.

        Raises:
            FileNotFoundError: If the path doesn't exist.

        Example:
            >>> # Create dataset
            >>> dataset1 = Dataset[str](path="/tmp/mydata")
            >>> dataset1.append("hello")
            >>> dataset1.flush()
            >>> dataset1.close()
            >>>
            >>> # Load dataset later
            >>> dataset2 = Dataset.from_path("/tmp/mydata")
            >>> dataset2[0]
            'hello'
        """
        return cls(path)
