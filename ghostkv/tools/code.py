"""Python code execution tool."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


class CodeTool:
    """Execute Python code in a subprocess.

    Args:
        timeout: Max execution time in seconds
        python_path: Path to Python interpreter
    """

    name = "run"

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        python_path: str = "python",
    ):
        self.timeout = timeout
        self.python_path = python_path

    def run(self, code: str) -> str:
        """Execute Python code and return stdout + stderr.

        Args:
            code: Python source code to execute

        Returns:
            Combined stdout and stderr output, truncated to 4000 chars.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self.python_path, tmp_path],
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
            return f"Timeout: code execution exceeded {self.timeout}s"
        except Exception as e:
            return f"Execution error: {e}"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
