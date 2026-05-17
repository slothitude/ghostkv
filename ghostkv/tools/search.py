"""Web search tool via SearXNG."""

from __future__ import annotations

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_SEARXNG_URL = "http://192.168.0.33:8888"


class SearchTool:
    """Search the web via a SearXNG instance.

    Args:
        base_url: SearXNG HTTP endpoint (default: Lappy:8888)
        timeout: Request timeout in seconds
    """

    name = "search"

    def __init__(
        self,
        base_url: str = DEFAULT_SEARXNG_URL,
        timeout: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def run(self, query: str, max_results: int = 5) -> str:
        """Search and return formatted results.

        Returns:
            Formatted string with title, URL, and snippet for each result.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"Search error: {e}"

        results = data.get("results", [])[:max_results]
        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

        return "\n\n".join(lines)
