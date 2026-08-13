"""Process-level output library locking for manifest writers."""

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def output_lock(output_dir: Path):
    lock_path = output_dir / ".phelper" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
