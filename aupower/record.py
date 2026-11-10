"""Records of what was run and what was sent: the invocation log and the API call log.

* ``invocations.jsonl`` (one line per command that runs stages): the command line, profile, overrides, trajectory
  selection, the fully resolved config, a copy of config.toml as it was (``configs/config.<time>.toml``), the
  interpreter, platform and package versions, and whether a key was present (never the key itself).
* ``api_calls.jsonl.gz`` (one line per HTTP attempt to OpenRouter): the exact request body, the raw response JSON
  or the error, status, latency, purpose and meta. Screenshot data URLs are replaced by their size and hash, since
  the screenshots themselves are saved in the rollout directories. The file is gzip-compressed (JSON lines inside):
  read it with ``gzip -dc runs/<name>/api_calls.jsonl.gz | jq`` or ``python -m aupower calls``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .config import API_KEY_ENV_VAR, Config, get_openrouter_api_key

API_LOG_NAME = "api_calls.jsonl.gz"
INVOCATIONS_NAME = "invocations.jsonl"


# --------------------------------------------------------------------------------------------
# API calls
# --------------------------------------------------------------------------------------------

def redact_images(messages: Any) -> Any:
    """A copy of a message list with every image data URL replaced by its mime type, size and sha256."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        parts = []
        for part in m["content"]:
            url = (part.get("image_url") or {}).get("url") if isinstance(part, dict) and part.get("type") == "image_url" else None
            if isinstance(url, str) and url.startswith("data:"):
                head, _, payload = url.partition(",")
                digest = hashlib.sha256(payload.encode("ascii", "ignore")).hexdigest()[:16]
                parts.append({"type": "image_url", "image_url": {"url": f"<{head} omitted: {len(payload)} chars, sha256 {digest}>"}})
            else:
                parts.append(part)
        out.append({**m, "content": parts})
    return out


class CallLog:
    """Append-only gzip JSONL of every API request and response. Writes are synchronous and small, so coroutines
    sharing the loop never interleave inside one line."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.n = 0

    def write(self, *, kind: str, request: dict[str, Any] | None, status: int | None, latency_s: float,
              response: Any = None, error: str | None = None, purpose: str | None = None, meta: dict | None = None,
              model: str | None = None, endpoint: str | None = None, attempt: int | None = None,
              request_id: str | None = None) -> None:
        entry = {"ts": time.time(), "ts_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "kind": kind,
                 "purpose": purpose, "meta": meta or {}, "model": model, "endpoint": endpoint, "attempt": attempt,
                 "request": request, "status": status, "latency_s": round(latency_s, 3), "response": response,
                 "error": error, "request_id": request_id}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.path, "at", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        self.n += 1


def read_calls(path: Path, *, purpose: str | None = None, kind: str | None = None, limit: int | None = None) -> list[dict]:
    """Read the API call log back (optionally filtered), newest last."""
    out: list[dict] = []
    p = Path(path)
    if not p.exists():
        return out
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if purpose and e.get("purpose") != purpose:
                continue
            if kind and e.get("kind") != kind:
                continue
            out.append(e)
    return out[-limit:] if limit else out


# --------------------------------------------------------------------------------------------
# Invocations
# --------------------------------------------------------------------------------------------

def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("httpx", "playwright"):
        try:
            from importlib.metadata import version
            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def _git_commit(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            target = root / ".git" / ref[5:]
            return target.read_text(encoding="utf-8").strip() if target.exists() else None
        return ref or None
    except Exception:
        return None


def log_invocation(cfg: Config, argv: list[str], *, stages: list[str] | None = None,
                   trajectories: list[str] | None = None, note: str | None = None) -> dict[str, Any]:
    """Append this command to runs/<name>/invocations.jsonl with the resolved config, and keep a copy of
    config.toml as it was. Returns the entry."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    config_copy = None
    src = Path(cfg.source_path)
    if src.exists():
        copies = run_dir / "configs"
        copies.mkdir(parents=True, exist_ok=True)
        config_copy = copies / f"config.{stamp}.toml"
        n = 1
        while config_copy.exists():
            n += 1
            config_copy = copies / f"config.{stamp}-{n}.toml"
        shutil.copyfile(src, config_copy)
    entry = {
        "ts": time.time(), "ts_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "argv": list(argv), "command": " ".join(["python -m aupower"] + list(argv)),
        "profile": cfg.profile, "mode": cfg.run.mode, "run_name": cfg.run.name, "stages": stages,
        "trajectories": list(trajectories or []) or None,
        "config_toml_copy": str(config_copy) if config_copy else None,
        "config_toml_sha256": hashlib.sha256(src.read_bytes()).hexdigest() if src.exists() else None,
        "config_resolved": cfg.to_dict(),
        "api_key": "present" if get_openrouter_api_key() else "absent",      # never the key itself
        "api_key_env_var": API_KEY_ENV_VAR,
        "python": sys.version.split()[0], "platform": platform.platform(), "hostname": platform.node(),
        "packages": _package_versions(), "git_commit": _git_commit(cfg.project_root),
        "cwd": os.getcwd(), "pid": os.getpid(), "note": note,
    }
    with open(run_dir / INVOCATIONS_NAME, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return entry
