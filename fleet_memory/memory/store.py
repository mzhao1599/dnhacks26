"""Append-only JSONL event log. Every state change is a new event; nothing is mutated in place.

Writers do one os.write() of one line on an O_APPEND fd, so worker processes can share a file
without a lock. Readers replay events to materialise current state (e.g. a lesson's status).
"""
from __future__ import annotations

import json
import os
from typing import Iterator

from fleet_memory.memory.schema import Episode, Lesson, Record, _clean, record_from_dict


class EventStore:
    def __init__(self, path: str = "logs/events.jsonl"):
        self.path = str(path)
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._cache_key: tuple | None = None       # (size, mtime_ns) of the file the cache was read from
        self._cache: list[dict] = []

    # ------------------------------------------------------------------ write
    def append(self, record: Record | dict) -> None:
        d = _clean(record)
        if not isinstance(d, dict):
            raise TypeError(f"record must be a Record or dict, got {type(record).__name__}")
        data = (json.dumps(d, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":   # heal a torn tail from a crashed writer
                data = b"\n" + data                          # (a racing extra blank line is harmless)
            n = os.write(fd, data)
            while n < len(data):                    # short writes don't happen on local files; be safe anyway
                n += os.write(fd, data[n:])
        finally:
            os.close(fd)

    # ------------------------------------------------------------------- read
    def read_all(self) -> list[dict]:
        """All events in file order. Torn lines (crashed writer) are skipped. Treat dicts as read-only."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return []
        key = (st.st_size, st.st_mtime_ns)
        if key != self._cache_key:
            out: list[dict] = []
            with open(self.path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(d, dict):
                        out.append(d)
            self._cache, self._cache_key = out, key
        return list(self._cache)

    def iter_type(self, t: str) -> Iterator[dict]:
        return (d for d in self.read_all() if d.get("type") == t)

    def lessons(self) -> dict[str, Lesson]:
        """Base 'lesson' events (last emission wins) with each lesson's latest 'lesson_status' applied."""
        out: dict[str, Lesson] = {}
        for d in self.iter_type("lesson"):
            try:
                out[d["lesson_id"]] = record_from_dict(d)
            except (KeyError, TypeError):
                continue                            # malformed lesson event; intake validation should prevent this
        latest: dict[str, dict] = {}
        for d in self.iter_type("lesson_status"):
            latest[d.get("lesson_id")] = d          # file order => last seen is latest
        for lid, d in latest.items():
            l = out.get(lid)
            if l is None:
                continue
            ch = record_from_dict(d)
            l.status = ch.to_status
            if ch.evidence is not None:
                l.evidence = ch.evidence
        return out

    def episodes(self) -> list[Episode]:
        out: list[Episode] = []
        for d in self.iter_type("episode"):
            try:
                out.append(record_from_dict(d))
            except (KeyError, TypeError):
                continue
        return out

    def snapshots(self) -> list[dict]:
        return list(self.iter_type("snapshot"))
