from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from typing import Any


def _face_suffix(
    algo: str,
    *,
    cbf_face: str,
    cf_item_face: str,
    cf_user_face: str,
) -> str:
    algo = str(algo).lower()
    if algo == "cbf":
        return str(cbf_face).lower()
    if algo == "cf_item":
        return str(cf_item_face).lower()
    if algo == "cf_user":
        return str(cf_user_face).lower()
    return ""


@dataclass(slots=True)
class SimulationLogBuffer:
    out_prefix: str
    display_algorithm: str
    cbf_face: str
    cf_item_face: str
    cf_user_face: str
    log_lines: list[str] = field(default_factory=list)
    _log_path: Path | None = None
    _log_fp: TextIO | None = None
    _written_count: int = 0

    @property
    def face_suffix(self) -> str:
        return _face_suffix(
            self.display_algorithm,
            cbf_face=self.cbf_face,
            cf_item_face=self.cf_item_face,
            cf_user_face=self.cf_user_face,
        )

    @property
    def out_prefix_face(self) -> str:
        suffix = self.face_suffix
        return f"{self.out_prefix}_{suffix}" if suffix else self.out_prefix

    @property
    def log_file_path(self) -> str:
        if self._log_path is not None:
            return str(self._log_path)
        return f"{self.out_prefix_face}_analysis.txt"

    def set_log_file_path(self, path: str | Path) -> str:
        new_path = Path(path).expanduser().resolve()
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
        self._log_path = new_path
        self._log_fp = new_path.open("a", encoding="utf-8")
        self._sync_backlog()
        return str(new_path)

    def _sync_backlog(self) -> None:
        if self._log_fp is None:
            return
        if self._written_count >= len(self.log_lines):
            return
        for msg in self.log_lines[self._written_count:]:
            self._log_fp.write(f"{msg}\n")
        self._log_fp.flush()
        self._written_count = len(self.log_lines)

    def log_line(self, msg: str) -> None:
        line = str(msg)
        self.log_lines.append(line)
        if self._log_fp is not None:
            self._log_fp.write(f"{line}\n")
            self._written_count += 1

    def log_and_print(self, msg: str, *, flush: bool = False) -> None:
        self.log_line(msg)
        if flush and self._log_fp is not None:
            self._log_fp.flush()
        print(msg, flush=flush)

    def close(self) -> None:
        self._sync_backlog()
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
            self._log_fp = None

def create_log_buffer(
    *,
    out_prefix: str,
    display_algorithm: str,
    cbf_face: str,
    cf_item_face: str,
    cf_user_face: str,
) -> SimulationLogBuffer:
    return SimulationLogBuffer(
        out_prefix=out_prefix,
        display_algorithm=display_algorithm,
        cbf_face=cbf_face,
        cf_item_face=cf_item_face,
        cf_user_face=cf_user_face,
    )


def setup_logging(
    *,
    out_prefix: str,
    display_algorithm: str,
    cbf_face: str,
    cf_item_face: str,
    cf_user_face: str,
    log_progress_every: int,
) -> dict[str, Any]:
    log_buffer = create_log_buffer(
        out_prefix=out_prefix,
        display_algorithm=display_algorithm,
        cbf_face=cbf_face,
        cf_item_face=cf_item_face,
        cf_user_face=cf_user_face,
    )
    return {
        "_LOG_BUFFER": log_buffer,
        "OUT_PREFIX_FACE": log_buffer.out_prefix_face,
        "LOG_FILE_PATH": log_buffer.log_file_path,
        "LOG_PROGRESS_EVERY": int(log_progress_every),
        "log_line": log_buffer.log_line,
        "log_and_print": log_buffer.log_and_print,
        "set_log_file_path": log_buffer.set_log_file_path,
        "close_log_file": log_buffer.close,
    }
