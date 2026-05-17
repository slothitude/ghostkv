"""Obsidian vault memory tool — markdown files with frontmatter and wikilinks."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VAULT_PATH = Path.home() / ".ghostkv" / "memory"


class MemoryTool:
    """Search and write to an Obsidian-compatible markdown vault.

    Vault layout:
        ~/.ghostkv/memory/
            *.md files with YAML frontmatter and wikilinks

    Each memory file:
        ---
        type: observation  # or fact, entity, task, reflection
        timestamp: 2026-05-17T14:30:00
        tags: [tag1, tag2]
        related: "[[Other Note]]"
        ---
        Content with [[wikilinks]] here.

    Args:
        vault_path: Path to the vault directory
    """

    name = "recall"

    def __init__(self, vault_path: Path | str | None = None):
        self.vault_path = Path(vault_path) if vault_path else DEFAULT_VAULT_PATH
        self.vault_path.mkdir(parents=True, exist_ok=True)

    def run(self, query: str, top_k: int = 5) -> str:
        """Search the vault for relevant memories.

        Args:
            query: Search query (matches against content and tags)
            top_k: Maximum results to return

        Returns:
            Formatted string with matching memory files.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored: list[tuple[int, Path, str]] = []

        for md_file in self.vault_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8", errors="replace")

            # Score by word overlap
            content_lower = content.lower()
            score = sum(1 for w in query_words if w in content_lower)

            # Boost for title match
            title = md_file.stem.lower().replace("_", " ")
            title_words = set(title.split())
            score += sum(2 for w in query_words if w in title_words)

            if score > 0:
                # Extract just the body (after frontmatter)
                body = self._strip_frontmatter(content)
                scored.append((score, md_file, body))

        if not scored:
            return "No memories found matching that query."

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, path, body in scored[:top_k]:
            title = path.stem.replace("_", " ")
            snippet = body.strip()[:500]
            results.append(f"## {title} (relevance: {score})\n{snippet}")

        return "\n\n---\n\n".join(results)

    def write(
        self,
        title: str,
        content: str,
        memory_type: str = "observation",
        tags: list[str] | None = None,
        related: str | None = None,
    ) -> str:
        """Write a new memory file to the vault.

        Args:
            title: Note title (used as filename)
            content: Markdown content (may include [[wikilinks]])
            memory_type: observation, fact, entity, task, or reflection
            tags: List of tags
            related: Related note name (wikilink)

        Returns:
            Path to the created file.
        """
        # Slugify title for filename
        slug = re.sub(r"[^\w\s-]", "", title.lower())
        slug = re.sub(r"[\s_]+", "_", slug).strip("_")[:80]
        if not slug:
            slug = f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        filepath = self.vault_path / f"{slug}.md"

        # Build frontmatter
        ts = datetime.now().isoformat(timespec="seconds")
        tags_yaml = str(tags or []) if tags else "[]"
        related_yaml = f'"[[{related}]]"' if related else ""

        frontmatter = f"---\ntype: {memory_type}\ntimestamp: {ts}\ntags: {tags_yaml}\n"
        if related_yaml:
            frontmatter += f"related: {related_yaml}\n"
        frontmatter += "---\n\n"

        filepath.write_text(frontmatter + content + "\n", encoding="utf-8")
        logger.info(f"Memory written: {filepath}")
        return str(filepath)

    def list_files(self) -> list[str]:
        """List all memory files in the vault."""
        return [f.stem.replace("_", " ") for f in self.vault_path.glob("*.md")]

    def count(self) -> int:
        """Count memory files in the vault."""
        return len(list(self.vault_path.glob("*.md")))

    def _strip_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                return content[end + 3:]
        return content
