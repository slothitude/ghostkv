"""File read and write tools."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_READ_SIZE = 100_000  # 100KB


class FileReadTool:
    """Read a file from the local filesystem."""

    name = "read"

    def run(self, path: str) -> str:
        """Read file contents.

        Args:
            path: File path (absolute or relative to cwd)

        Returns:
            File contents as string, truncated to 100KB.
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"File not found: {p}"
        if not p.is_file():
            return f"Not a file: {p}"
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            if len(content) > MAX_READ_SIZE:
                content = content[:MAX_READ_SIZE] + f"\n... (truncated, {len(content)} total chars)"
            return content
        except Exception as e:
            return f"Read error: {e}"


class FileWriteTool:
    """Write content to a file on the local filesystem."""

    name = "write"

    def run(self, path: str, content: str) -> str:
        """Write content to a file.

        Args:
            path: File path
            content: Content to write

        Returns:
            Confirmation message with bytes written.
        """
        p = Path(path).expanduser().resolve()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {p}"
        except Exception as e:
            return f"Write error: {e}"
