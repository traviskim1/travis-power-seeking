"""A minimal PNG writer (no third-party dependency), used for mock screenshots in dry runs."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path | str, width: int, height: int, rgb: tuple[int, int, int],
              stripe: tuple[int, int, int] | None = None) -> Path:
    """Write a solid-colour PNG (with an optional horizontal stripe) of the given size."""
    path = Path(path)
    rows = []
    for y in range(height):
        colour = stripe if (stripe is not None and height // 3 <= y < 2 * height // 3) else rgb
        rows.append(b"\x00" + bytes(colour) * width)
    raw = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", zlib.compress(raw, 6))
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)
    return path
