from __future__ import annotations

from pathlib import Path
from typing import TextIO


class PrefixedLogger:
    """
    Simple logger that writes to stdout and a file with prefixed, lower-case messages.
    stdout logging can be disabled (useful for nonzero DDP ranks).
    """

    def __init__(self, log_path: Path, write_stdout: bool = True) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path: Path = log_path
        self._fh: TextIO = log_path.open("a", encoding="utf-8")
        self.write_stdout: bool = write_stdout

    def log(self, message: str, prefix: str = "[+]") -> None:
        safe_message: str = message.lower()
        line: str = f"{prefix} {safe_message}"
        if self.write_stdout:
            print(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
