from __future__ import annotations

from dataclasses import dataclass, field
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
        return f"{self.out_prefix_face}_analysis.txt"

    def log_line(self, msg: str) -> None:
        self.log_lines.append(str(msg))

    def log_and_print(self, msg: str, *, flush: bool = False) -> None:
        self.log_line(msg)
        print(msg, flush=flush)

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
    }
