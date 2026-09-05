"""Lesson retrieval: situation text -> embedding -> filtered, similarity-ranked lessons.

Retrieval is frozen pre-episode: the worker calls Retriever.retrieve() once before env.reset()
and writes the ids into Episode.retrieved_lesson_ids. Nothing here touches an env or an LLM.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable

import numpy as np

from fleet_memory.envs.base import TaskInfo
from fleet_memory.memory.schema import Lesson

_TOKEN = re.compile(r"[a-z0-9]+")


def situation_text(task: TaskInfo, instruction: str, phase: str = "*") -> str:
    """What gets embedded: task + scene objects + phase + instruction. Also stored on Snapshot."""
    objs = ", ".join(task.objects) if task.objects else "none"
    return f"{task.suite} {task.task_family} task {task.task_id}: {instruction} | objects: {objs} | phase: {phase}"


class Embedder:
    """sentence-transformers 'all-MiniLM-L6-v2' when importable and FM_EMBEDDER != 'hash';
    otherwise a deterministic hashed bag-of-words (unigrams + bigrams, signed, dim 256)."""

    def __init__(self, dim: int = 256, model_name: str = "all-MiniLM-L6-v2"):
        self._st = None
        if os.environ.get("FM_EMBEDDER", "") != "hash":
            try:
                from sentence_transformers import SentenceTransformer  # optional dep (pyproject [sim])
                self._st = SentenceTransformer(model_name)
            except Exception:
                self._st = None
        self.backend = "st" if self._st is not None else "hash"
        self.dim = int(self._st.get_sentence_embedding_dimension()) if self._st is not None else int(dim)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = _TOKEN.findall(text.lower())
        return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]

    def embed(self, text: str) -> np.ndarray:
        """Unit-norm float32 vector (all-zeros for empty text)."""
        text = text or ""
        if self._st is not None:
            v = np.asarray(self._st.encode(text, normalize_embeddings=True), dtype=np.float32)
        else:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in self._tokens(text):
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                v[int.from_bytes(h[:4], "little") % self.dim] += 1.0 if h[4] & 1 else -1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def lesson_text(lesson: Lesson) -> str:
    """The text a lesson is indexed under: the situation it was learned in, else its rationale."""
    prov = lesson.provenance or {}
    return str(prov.get("situation_text") or lesson.rationale_text or "")


class Retriever:
    def __init__(self, store, embedder: Embedder | None = None, k: int | None = 5,
                 randomize: bool = False, seed: int = 0):
        self.store = store
        self.embedder = embedder or Embedder()
        self.k = k
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)
        self.last_scores: dict[str, float] = {}     # lesson_id -> cosine, from the last retrieve() (debug/dashboard)
        self._cache: dict[str, np.ndarray] = {}     # text -> embedding

    def _embed(self, text: str) -> np.ndarray:
        v = self._cache.get(text)
        if v is None:
            v = self._cache[text] = self.embedder.embed(text)
        return v

    @staticmethod
    def is_eligible(lesson: Lesson, task: TaskInfo, surfaces_enabled: Iterable[str],
                    statuses: Iterable[str]) -> bool:
        t = lesson.trigger
        return (
            lesson.status in set(statuses)
            and lesson.surface in set(surfaces_enabled)
            and t.task_family in (task.task_family, "*")
            and (t.object_class == "*" or any(t.object_class in o for o in task.objects))
        )

    def retrieve(self, task: TaskInfo, surfaces_enabled: list[str],
                 statuses: Iterable[str] = ("candidate", "validated")) -> list[Lesson]:
        """Top-k eligible lessons by cosine(situation, lesson text). randomize=True (arm E): uniform
        random k of ALL lessons in `statuses`, ignoring eligibility and similarity."""
        statuses = tuple(statuses)
        lessons = list(self.store.lessons().values())
        self.last_scores = {}
        if self.randomize:
            pool = [l for l in lessons if l.status in statuses]
            if self.k is not None and len(pool) > self.k:
                idx = self.rng.choice(len(pool), size=self.k, replace=False)
                pool = [pool[i] for i in sorted(idx)]
            return pool
        q = self._embed(situation_text(task, task.language))
        scored = []
        for l in lessons:
            if not self.is_eligible(l, task, surfaces_enabled, statuses):
                continue
            s = cosine(q, self._embed(lesson_text(l)))
            self.last_scores[l.lesson_id] = s
            scored.append((-s, l.lesson_id, l))
        scored.sort(key=lambda x: (x[0], x[1]))     # deterministic tie-break on lesson_id
        out = [l for _, _, l in scored]
        return out if self.k is None else out[: self.k]
