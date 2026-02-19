from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BatchJob:
    display_algorithm: str
    face_name: str | None = None
    face_value: str | None = None

    @property
    def suffix(self) -> str:
        if self.face_name and self.face_value:
            return f"{self.display_algorithm}_{self.face_value}"
        return self.display_algorithm


JOBS: tuple[BatchJob, ...] = (
    BatchJob("random"),
    BatchJob("cbf", "CBF_FACE", "affinity"),
    BatchJob("cbf", "CBF_FACE", "novelty"),
    BatchJob("cf_user", "CF_USER_FACE", "affinity"),
    BatchJob("cf_user", "CF_USER_FACE", "novelty"),
    BatchJob("cf_item", "CF_ITEM_FACE", "affinity"),
    BatchJob("cf_item", "CF_ITEM_FACE", "novelty"),
    BatchJob("popularity"),
    BatchJob("trend"),
    BatchJob("buzz"),
)


def _run_job(job: BatchJob, *, python_exe: str) -> int:
    env = os.environ.copy()
    env["DISPLAY_ALGORITHM"] = job.display_algorithm
    if job.face_name and job.face_value:
        env[job.face_name] = job.face_value

    base_prefix = env.get("OUT_PREFIX_BASE") or env.get("OUT_PREFIX")
    if base_prefix:
        env["OUT_PREFIX"] = f"{base_prefix}_{job.suffix}"
    else:
        env["OUT_PREFIX"] = f"simulation_{job.suffix}"

    main_script = Path(__file__).resolve().parent / "main.py"
    cmd = [python_exe, str(main_script)]
    print(f"==== start: {job.display_algorithm} {job.face_name or ''} {job.face_value or ''} ====")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode == 0:
        print(f"==== done : {job.suffix} ====")
    else:
        print(f"!!!! fail : {job.suffix} (rc={proc.returncode})")
    return int(proc.returncode)


def main() -> int:
    print("[batch] run all algorithms")
    print(
        "[batch] env passthrough: MAX_STEPS / NUM_AGENTS / NUM_CONTENTS / "
        "RANDOM_SEED / STRICT_CUDA / OUT_PREFIX_BASE ..."
    )
    for job in JOBS:
        rc = _run_job(job, python_exe=sys.executable)
        if rc != 0:
            return rc
    print("[batch] all jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
