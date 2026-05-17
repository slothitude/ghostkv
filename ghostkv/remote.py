"""Remote API backend — OpenAI-compatible chat completion client.

Parallel to model.ModelBackend — does NOT implement that ABC (too tensor-specific).
Instead, provides RemoteBackend.generate() for the agent's remote path and
MessageSession as a duck-type replacement for KVSession.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RemoteBackend — OpenAI-compatible chat completion
# ---------------------------------------------------------------------------

class RemoteBackend:
    """Synchronous OpenAI-compatible chat completion client.

    Returns (text, usage) so the agent can track token costs from the API response.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "glm-5.1",
        timeout: int = 120,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> tuple[str, dict]:
        """Send a chat completion request and return (text, usage_stats).

        Uses retry on 429 with backoff (Alfred pattern: 10s, 20s, 30s).
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._post_with_retry(headers, payload)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Remote API error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, usage

    def _post_with_retry(
        self,
        headers: dict,
        payload: dict,
        retries: int = 3,
    ) -> requests.Response:
        """POST with retry on 429 rate limiting."""
        for i in range(retries):
            resp = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 429 and i < retries - 1:
                wait = 10 * (i + 1)
                logger.warning("Rate limited, waiting %ds...", wait)
                time.sleep(wait)
                continue
            return resp
        return resp


# ---------------------------------------------------------------------------
# MessageSession — duck-type replacement for KVSession
# ---------------------------------------------------------------------------

class MessageSession:
    """Message-based session for remote backends.

    Mirrors KVSession interface: save(), load(), reset(), log(), stats().
    Uses messages.json instead of kv_cache.bin.
    """

    def __init__(
        self,
        name: str = "default",
        model_name: str = "unknown",
    ):
        self.name = name
        self.model_name = model_name
        self.base_dir = Path.home() / ".ghostkv" / "sessions" / name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state
        self._messages: list[dict] = []
        self.steps: int = 0
        self.total_tokens: int = 0
        self.token_costs: list[int] = []
        self._log_lines: list[str] = []

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    @property
    def messages_path(self) -> Path:
        return self.base_dir / "messages.json"

    @property
    def metadata_path(self) -> Path:
        return self.base_dir / "metadata.json"

    @property
    def log_path(self) -> Path:
        return self.base_dir / "conversation.log"

    def add_message(self, role: str, content: str):
        """Append a message to the conversation history."""
        self._messages.append({"role": role, "content": content})

    def save(self):
        """Serialize messages and metadata to disk."""
        meta = {
            "model": self.model_name,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "token_costs": self.token_costs,
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        if self._messages:
            self.messages_path.write_text(
                json.dumps(self._messages, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Messages saved: %d messages", len(self._messages))

        # Append log
        if self._log_lines:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.writelines(self._log_lines)
            self._log_lines.clear()

    def load(self, device: str = "cpu") -> bool:
        """Load messages and metadata from disk.

        Returns True if session was loaded, False if empty/new.
        """
        if not self.metadata_path.exists():
            logger.info("New session: %s", self.name)
            return False

        meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model_name = meta.get("model", self.model_name)
        self.steps = meta.get("steps", 0)
        self.total_tokens = meta.get("total_tokens", 0)
        self.token_costs = meta.get("token_costs", [])

        if self.messages_path.exists():
            raw = self.messages_path.read_text(encoding="utf-8")
            self._messages = json.loads(raw)
            logger.info("Messages loaded: %d messages", len(self._messages))

        return True

    def reset(self):
        """Clear all state, start fresh."""
        self._messages.clear()
        self.steps = 0
        self.total_tokens = 0
        self.token_costs = []
        self._log_lines.clear()
        logger.info("Session reset.")

    def log(self, line: str):
        """Append a line to the conversation log."""
        self._log_lines.append(line + "\n")

    def stats(self) -> dict:
        """Return session statistics."""
        return {
            "session": self.name,
            "model": self.model_name,
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_step": self.total_tokens / max(1, self.steps),
            "messages": len(self._messages),
            "mode": "remote",
        }
