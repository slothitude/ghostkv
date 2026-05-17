"""Bash shell command execution tool."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class BashTool:
    """Execute bash commands in a subprocess.

    Args:
        timeout: Max execution time in seconds
        shell: Shell executable (default: bash)
    """

    name = "bash"

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        shell: str = "bash",
    ):
        self.timeout = timeout
        self.shell = shell

    def run(self, command: str) -> str:
        """Execute a bash command and return stdout + stderr.

        Args:
            command: Shell command to execute

        Returns:
            Combined stdout and stderr output, truncated to 4000 chars.
        """
        try:
            result = subprocess.run(
                [self.shell, "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
            if not output:
                output = "(no output)"

            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"

            return output[:4000]

        except subprocess.TimeoutExpired:
            return f"Timeout: command exceeded {self.timeout}s"
        except FileNotFoundError:
            return f"Shell not found: {self.shell}"
        except Exception as e:
            return f"Execution error: {e}"
