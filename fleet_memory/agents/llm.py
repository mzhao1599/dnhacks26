"""LLM client for the planner and the two coaches, plus the deterministic offline mock.

One call shape only: `complete_json(system, user, schema)` -> dict. Backends, chosen by
`FM_LLM` or by which key is present:
  - anthropic : one forced tool call on the Messages API (input_schema=schema)       [ANTHROPIC_API_KEY]
  - gemini    : generate_content with response_mime_type=application/json + schema   [GEMINI_API_KEY | GOOGLE_API_KEY]
  - mock      : MOCK_RESPONDERS[schema["title"]], registered by planner.py / coach_*.py at import time
Responders only see the user string, so callers embed their structured context with
`with_context` and responders read it back with `parse_context`.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import struct
import time
import zlib
from typing import Any, Callable

import numpy as np

from fleet_memory.execution.constraints import ConstraintSet, apply_edit
from fleet_memory.memory.schema import EDIT_OPS, PREDICATES, Edit, Lesson, _clean

PLANNER_MODEL = "claude-haiku-4-5-20251001"
COACH_MODEL = "claude-sonnet-5"

# Role-equivalent Gemini models, overridable with FM_PLANNER_MODEL / FM_COACH_MODEL.
GEMINI_MODELS = {
    PLANNER_MODEL: os.environ.get("FM_PLANNER_MODEL", "gemini-3.7-flash"),
    COACH_MODEL: os.environ.get("FM_COACH_MODEL", "gemini-3.1-pro-preview"),   # 2.5-pro is 404 for new keys
}

log = logging.getLogger(__name__)


def pick_backend() -> str:
    """FM_LLM wins; else the first key found; else mock."""
    b = os.environ.get("FM_LLM", "").lower()
    if b in ("mock", "anthropic", "gemini"):
        return b
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    return "mock"

# schema title -> fn(user_text) -> dict. Populated by planner.py / coach_*.py.
MOCK_RESPONDERS: dict[str, Callable[[str], dict]] = {}


def register_mock(title: str):
    def deco(fn):
        MOCK_RESPONDERS[title] = fn
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Context block: structured state embedded in the user turn
# --------------------------------------------------------------------------- #

_CTX_RE = re.compile(r"<context>\s*(\{.*\})\s*</context>", re.S)


def with_context(text: str, ctx: dict) -> str:
    """Append a JSON context block the model (and the mock) can read."""
    body = json.dumps(_clean(ctx), indent=1, default=str)
    return f"{text}\n\n<context>\n{body}\n</context>"


def parse_context(user: str) -> dict:
    m = _CTX_RE.search(user)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def _png_bytes(img: np.ndarray) -> bytes:
    """Minimal RGB8 PNG encoder (no PIL dependency)."""
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, :3].astype(np.uint8)
    h, w = img.shape[:2]
    rows = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * 3)], axis=1)  # filter byte 0 per row

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + chunk(b"IEND", b""))


def image_bytes(img: np.ndarray) -> tuple[bytes, str]:
    """HxWx3 uint8 -> (encoded bytes, media type): jpeg via PIL if available, else png."""
    try:
        from PIL import Image  # optional
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(img[:, :, :3]).astype(np.uint8)).save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return _png_bytes(img), "image/png"


def image_block(img: np.ndarray) -> dict:
    """HxWx3 uint8 -> anthropic Messages API image content block."""
    data, media = image_bytes(img)
    return {"type": "image", "source": {"type": "base64", "media_type": media,
                                        "data": base64.standard_b64encode(data).decode("ascii")}}


def subsample(frames: list, n: int) -> list:
    """At most n frames, evenly spaced, always keeping the last one."""
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded", "rate_limit", "DEADLINE_EXCEEDED")


class LLM:
    def __init__(self, model: str, mock: bool | None = None, backend: str | None = None):
        self.backend = "mock" if mock else (backend or pick_backend())
        if mock is False and self.backend == "mock":
            raise RuntimeError("mock=False but no API key found (set ANTHROPIC_API_KEY or GEMINI_API_KEY)")
        self.mock = self.backend == "mock"
        # `model` is always given as the Claude role name; map to the Gemini equivalent if needed.
        self.model = GEMINI_MODELS.get(model, model) if self.backend == "gemini" else model
        self._client = None
        self.max_retries = int(os.environ.get("FM_LLM_RETRIES", "5"))

    def complete_json(self, system: str, user: str, schema: dict,
                      images: list[np.ndarray] | None = None, max_tokens: int = 2000) -> tuple[dict, dict]:
        """Returns (output_dict, meta) with meta = {tokens_in, tokens_out, latency_s, model}."""
        t0 = time.time()
        if self.mock:
            responder = MOCK_RESPONDERS.get(schema.get("title", ""))
            if responder is None:
                raise KeyError(f"no mock responder for schema title {schema.get('title')!r}")
            out = responder(user)
            meta = {"tokens_in": (len(system) + len(user)) // 4 + 1000 * len(images or []),
                    "tokens_out": len(json.dumps(out, default=str)) // 4,
                    "latency_s": time.time() - t0, "model": f"mock:{self.model}"}
            return out, meta

        call = self._anthropic if self.backend == "anthropic" else self._gemini
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                out, tin, tout, model = call(system, user, schema, images or [], max_tokens)
                break
            except Exception as ex:  # rate limits / transient 5xx: back off, everything else raises
                if attempt >= self.max_retries or not any(k in repr(ex) for k in _RETRYABLE):
                    raise
                log.warning("%s call failed (%s); retry %d in %.0fs", self.backend, repr(ex)[:120], attempt + 1, delay)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return out, {"tokens_in": tin, "tokens_out": tout, "latency_s": time.time() - t0, "model": model}

    # --- anthropic --------------------------------------------------------- #
    def _anthropic(self, system, user, schema, images, max_tokens):
        content: list[dict[str, Any]] = [image_block(im) for im in images]
        content.append({"type": "text", "text": user})
        resp = self._get_client().messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
            tools=[{"name": "emit", "description": "Emit the structured result.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "emit"},
        )
        out = next((dict(b.input) for b in resp.content if b.type == "tool_use"), None)
        if out is None:
            raise RuntimeError(f"no tool_use block in response (stop_reason={resp.stop_reason})")
        return out, resp.usage.input_tokens, resp.usage.output_tokens, resp.model

    # --- gemini ------------------------------------------------------------ #
    def _gemini(self, system, user, schema, images, max_tokens):
        from google.genai import types
        parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in (image_bytes(im) for im in images)]
        parts.append(types.Part.from_text(text=user))
        cfg = dict(system_instruction=system, response_mime_type="application/json",
                   max_output_tokens=max_tokens, temperature=0.2)
        client = self._get_client()
        try:  # full JSON-schema support (newer SDKs); fall back to the OpenAPI-subset Schema
            resp = client.models.generate_content(
                model=self.model, contents=parts,
                config=types.GenerateContentConfig(response_json_schema=schema, **cfg))
        except (TypeError, ValueError):
            resp = client.models.generate_content(
                model=self.model, contents=parts,
                config=types.GenerateContentConfig(response_schema=gemini_schema(schema), **cfg))
        text = resp.text
        if not text:
            raise RuntimeError(f"empty gemini response: {getattr(resp, 'prompt_feedback', None)}")
        out = json.loads(text)
        u = getattr(resp, "usage_metadata", None)
        tin = int(getattr(u, "prompt_token_count", 0) or 0)
        tout = int(getattr(u, "candidates_token_count", 0) or 0) + int(getattr(u, "thoughts_token_count", 0) or 0)
        return out, tin, tout, self.model

    def _get_client(self):
        if self._client is None:
            if self.backend == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            else:
                from google import genai
                self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        return self._client


def gemini_schema(schema: Any) -> Any:
    """Strip JSON-schema keywords Gemini's OpenAPI-subset `response_schema` rejects; map null unions to nullable."""
    if isinstance(schema, list):
        return [gemini_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("title", "$schema", "additionalProperties", "default", "examples", "const"):
            continue
        if k == "type" and isinstance(v, list):
            non_null = [t for t in v if t != "null"]
            out["type"] = non_null[0] if non_null else "string"
            if "null" in v:
                out["nullable"] = True
            continue
        if k in ("anyOf", "oneOf") and isinstance(v, list):
            non_null = [s for s in v if s.get("type") != "null"]
            if len(non_null) == 1:
                out.update(gemini_schema(non_null[0]))
                if len(non_null) != len(v):
                    out["nullable"] = True
                continue
            out[k] = [gemini_schema(s) for s in non_null]
            continue
        out[k] = gemini_schema(v)
    return out


def list_gemini_models() -> list[str]:
    """Models the configured key can use for generateContent (for picking FM_*_MODEL)."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return sorted(m.name.removeprefix("models/") for m in client.models.list()
                  if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"]))


# --------------------------------------------------------------------------- #
# Lesson validation shared by both coaches
# --------------------------------------------------------------------------- #

def _local_validate(l: Lesson | dict) -> tuple[bool, str]:
    """Same rules as memory/lifecycle.validate_lesson; used only if that module is unavailable."""
    d = l if isinstance(l, dict) else l.to_dict()
    s, e, t = d.get("surface"), d.get("edit") or {}, d.get("trigger") or {}
    if s not in EDIT_OPS or s not in ("S1", "S2", "S3"):
        return False, f"bad surface {s!r}"
    if e.get("op") not in EDIT_OPS[s]:
        return False, f"op {e.get('op')!r} not in EDIT_OPS[{s}]"
    extra = set(e.get("params") or {}) - set(EDIT_OPS[s][e["op"]])
    if extra:
        return False, f"unknown params {sorted(extra)} for {e['op']}"
    if t.get("predicate") not in PREDICATES:
        return False, f"bad trigger predicate {t.get('predicate')!r}"
    if s == "S3":
        try:
            apply_edit(ConstraintSet(), Edit(op=e["op"], params=dict(e.get("params") or {})))
        except Exception as ex:
            return False, f"S3 edit rejected: {ex}"
    return True, "ok"


def validate_lesson(l: Lesson | dict) -> tuple[bool, str]:
    """lifecycle.validate_lesson (imported lazily; memory module may be absent), else the local
    equivalent. Additionally any `predicate` param must itself be in PREDICATES."""
    try:
        from fleet_memory.memory.lifecycle import validate_lesson as _v
    except Exception:
        _v = _local_validate
    try:
        ok, why = _v(l)
    except Exception as ex:
        return False, f"validator raised: {ex}"
    if not ok:
        return False, why
    d = l if isinstance(l, dict) else l.to_dict()
    p = (d.get("edit") or {}).get("params") or {}
    if "predicate" in p and p["predicate"] not in PREDICATES:
        return False, f"unknown predicate in params: {p['predicate']!r}"
    return True, why


EDIT_OPS_TEXT = json.dumps(EDIT_OPS, indent=1)
PREDICATES_TEXT = "\n".join(f"- {p}" for p in PREDICATES)
