"""HTTP request tool."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15


class HttpTool:
    """Make HTTP requests.

    Args:
        timeout: Request timeout in seconds
    """

    name = "http"

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def run(
        self,
        method: str = "GET",
        url: str = "",
        body: str | None = None,
        headers: dict | None = None,
    ) -> str:
        """Make an HTTP request and return the response.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            url: Target URL
            body: Request body (for POST/PUT)
            headers: Optional headers dict

        Returns:
            Response body (truncated to 4000 chars) with status code.
        """
        if not url:
            return "Error: no URL provided"

        method = method.upper()
        try:
            resp = requests.request(
                method=method,
                url=url,
                data=body,
                headers=headers,
                timeout=self.timeout,
            )
            result = f"Status: {resp.status_code}\n"
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                result += resp.text[:4000]
            else:
                result += resp.text[:4000]
            return result
        except requests.RequestException as e:
            return f"HTTP error: {e}"
