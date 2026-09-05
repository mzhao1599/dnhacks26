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
        self._offset = 0                           # bytes of the file already parsed into _cache (append-only => resume here)
        self._episodes: list[Episode] = []         # hydrated once per record; extended incrementally
        self._episodes_scanned = 0                 # how many records of _cache have been scanned for episodes

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
            if st.st_size < self._offset:          # truncated/replaced: start over
                self._cache, self._offset = [], 0
                self._episodes, self._episodes_scanned = [], 0
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                data = f.read()
            end = data.rfind(b"\n") + 1            # only consume complete lines; a torn tail is re-read next time
            for line in data[:end].split(b"\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    self._cache.append(d)
            self._offset += end
            self._cache_key = key
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
        """Hydrated Episode records, in file order. Each record is hydrated once (append-only log)."""
        recs = self.read_all()
        for d in recs[self._episodes_scanned:]:
            if d.get("type") == "episode":
                try:
                    self._episodes.append(record_from_dict(d))
                except (KeyError, TypeError):
                    continue
        self._episodes_scanned = len(recs)
        return list(self._episodes)

    def snapshots(self) -> list[dict]:
        return list(self.iter_type("snapshot"))
