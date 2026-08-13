import multiprocessing
import time

from frame.locking import output_lock


def _hold_lock(output, ready):
    with output_lock(output):
        ready.set()
        time.sleep(0.3)


def test_output_lock_serializes_manifest_writers(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_lock, args=(tmp_path, ready))
    process.start()
    try:
        assert ready.wait(5)
        started = time.monotonic()
        with output_lock(tmp_path):
            waited = time.monotonic() - started
        assert waited >= 0.2
    finally:
        process.join(5)
        if process.is_alive():
            process.terminate()
    assert process.exitcode == 0
