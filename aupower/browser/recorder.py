"""Server layer (SPEC.md section 4): record every HTTP response the websites gave during a trajectory, and
replay the recording to forks. Requests the recording does not cover go to the live web and are logged,
so the live-fallback fraction can be reported per rollout."""
from __future__ import annotations

import base64
import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def request_key(method: str, url: str, post_data: str | bytes | None) -> tuple[str, str, str]:
    url = url.split("#", 1)[0]
    if post_data is None:
        ph = ""
    else:
        raw = post_data if isinstance(post_data, bytes) else str(post_data).encode("utf-8", "replace")
        ph = hashlib.sha1(raw).hexdigest()
    return (method.upper(), url, ph)


def _split_post(post_data: str | bytes | None) -> tuple[str | None, str | None]:
    """A POST body as it is stored: text in the historical `post` field when it is valid UTF-8, else base64 in `post_b64`."""
    if post_data is None:
        return None, None
    if isinstance(post_data, bytes):
        try:
            return post_data.decode("utf-8"), None
        except UnicodeDecodeError:
            return None, base64.b64encode(post_data).decode("ascii")
    return str(post_data), None


def entry_post(entry: dict) -> str | bytes | None:
    """The POST body of a recorded entry: text, or the raw bytes of a binary body."""
    b = entry.get("post_b64")
    if b is not None:
        return base64.b64decode(b)
    return entry.get("post")


class Recorder:
    """Appends response entries to a JSONL file as they arrive. ``count`` is the number recorded so far,
    which a snapshot stores so that a fork replays only what happened before the interrupt."""

    def __init__(self, path: Path, max_body_bytes: int = 10_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_body_bytes = max_body_bytes
        self.count = 0
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self.count = sum(1 for line in f if line.strip())
        self._fh = open(self.path, "a", encoding="utf-8")

    def add(self, *, method: str, url: str, status: int, headers: dict[str, str], body: bytes | None,
            post_data: str | bytes | None = None, mime: str = "") -> int:
        if body is not None and len(body) > self.max_body_bytes:
            body = None
        post_text, post_b64 = _split_post(post_data)
        entry = {
            "i": self.count, "t": time.time(), "method": method.upper(), "url": url, "status": int(status),
            "headers": {str(k): str(v) for k, v in headers.items()}, "mime": mime, "post": post_text,
            "body_b64": base64.b64encode(body).decode("ascii") if body is not None else None,
            "body_len": len(body) if body is not None else None,
        }
        if post_b64 is not None:
            entry["post_b64"] = post_b64
        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1
        return entry["i"]

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


@dataclass
class ReplayStats:
    served: int = 0
    live: int = 0
    live_urls: list[str] = field(default_factory=list)
    served_urls: list[str] = field(default_factory=list)
    max_logged: int = 200

    @property
    def total(self) -> int:
        return self.served + self.live

    @property
    def live_fraction(self) -> float | None:
        return (self.live / self.total) if self.total else None

    def log_served(self, url: str) -> None:
        self.served += 1
        if len(self.served_urls) < self.max_logged:
            self.served_urls.append(url)

    def log_live(self, url: str) -> None:
        self.live += 1
        if len(self.live_urls) < self.max_logged:
            self.live_urls.append(url)

    def to_dict(self) -> dict[str, Any]:
        return {"requests_total": self.total, "served_from_recording": self.served, "live_fallback": self.live,
                "live_fallback_fraction": self.live_fraction, "live_urls": list(self.live_urls)}


class ReplayIndex:
    """The recording up to a given entry, indexed for lookup by (method, url, post body)."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self._queues: dict[tuple, deque] = {}
        self._last: dict[tuple, dict] = {}
        for e in entries:
            k = request_key(e["method"], e["url"], entry_post(e))
            self._queues.setdefault(k, deque()).append(e)
            self._last[k] = e
        self.stats = ReplayStats()

    @classmethod
    def load(cls, path: Path, upto: int | None = None) -> "ReplayIndex":
        entries: list[dict] = []
        path = Path(path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if upto is not None and e["i"] >= upto:
                        break
                    entries.append(e)
        return cls(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def has(self, method: str, url: str, post_data: str | bytes | None = None) -> bool:
        return request_key(method, url, post_data) in self._last

    def take(self, method: str, url: str, post_data: str | bytes | None = None) -> dict | None:
        """The next recorded response for this request, in recorded order; the last one repeats."""
        k = request_key(method, url, post_data)
        q = self._queues.get(k)
        if q:
            return q.popleft()
        return self._last.get(k)

    @staticmethod
    def body_bytes(entry: dict) -> bytes | None:
        b = entry.get("body_b64")
        return base64.b64decode(b) if b is not None else None
