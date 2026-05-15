from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


def should_process(audio: Path, overwrite: bool) -> bool:
    sidecar = audio.with_suffix(".json")
    if overwrite or not sidecar.exists():
        return True
    return audio.stat().st_mtime > sidecar.stat().st_mtime


class WatchQueue:
    def __init__(self, debounce_seconds: float = 2.0) -> None:
        self._debounce = debounce_seconds
        self._pending: "OrderedDict[Path, float]" = OrderedDict()
        self._lock = threading.Lock()

    def enqueue(self, path: Path) -> None:
        with self._lock:
            self._pending[path] = time.monotonic()
            self._pending.move_to_end(path)

    def drain_ready(self) -> list[Path]:
        now = time.monotonic()
        ready: list[Path] = []
        with self._lock:
            for path, ts in list(self._pending.items()):
                if now - ts >= self._debounce:
                    ready.append(path)
                    self._pending.pop(path, None)
        return ready


class _Handler(FileSystemEventHandler):
    def __init__(self, queue: WatchQueue, extensions: set[str]) -> None:
        self._queue = queue
        self._exts = {e.lower().lstrip(".") for e in extensions}

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._maybe(event)

    def _maybe(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower().lstrip(".") in self._exts:
            self._queue.enqueue(p)


def run_watch(
    directory: Path,
    extensions: list[str],
    debounce_seconds: float,
    recursive: bool,
    on_ready: Callable[[Path], None],
    stop_event: threading.Event | None = None,
    poll_interval: float = 0.5,
) -> None:
    stop = stop_event or threading.Event()
    queue = WatchQueue(debounce_seconds=debounce_seconds)
    observer = Observer()
    observer.schedule(_Handler(queue, set(extensions)), str(directory), recursive=recursive)
    observer.start()
    try:
        while not stop.is_set():
            for path in queue.drain_ready():
                on_ready(path)
            time.sleep(poll_interval)
    finally:
        observer.stop()
        observer.join()
