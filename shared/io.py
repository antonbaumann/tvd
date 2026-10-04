"""Artifact I/O and local JSONL experiment events. No cross-run reporting."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

RUN_CONTEXT: dict[str, Any] = {}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def write_json(path, value):
    """JSON is reserved for functional inputs/manifests/checkpoint metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(clean(row), sort_keys=True, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_record(row):
    if row['event'] in {'eval_summary', 'eval_task', 'eval_trait', 'eval_example', 'eval_judge_generation'}:
        for key in ('phase', 'step', 'sequence', 'targeted_task'):
            row[key]
    return clean({'schema_version': 1, **RUN_CONTEXT, **row})


class JsonlLogger:
    def __init__(self, path: Path, enabled=True, append=False):
        self.handle = None
        self.enabled = enabled
        self.context = {}
        if enabled:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, row):
        if self.handle is not None:
            self.handle.write(json.dumps(event_record({**self.context, **row}), sort_keys=True, allow_nan=False) + "\n")
            self.handle.flush()

    def close(self):
        if self.handle:
            self.handle.close()


def write_run_record(output_dir, event, value):
    logger = JsonlLogger(Path(output_dir) / 'metadata.jsonl', append=True)
    logger.write({'event': event, {'resolved_config': 'config', 'run_complete': 'details'}[event]: value})
    logger.close()


def prepare_output_dir(path, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"{path} is not empty; choose a new run ID or use --overwrite")
        # Keep previous runs recoverable instead of deleting them.
        import time
        path.rename(path.with_name(path.name + f".previous-{time.time_ns()}"))
    path.mkdir(parents=True, exist_ok=True)


def evaluation_phase(state, step):
    if step == 0 and state in ('pre_segment', 'initial'):
        return 'initialization'
    return {'pre_segment': 'before_adaptation', 'pre_merge': 'local_endpoint',
            'post_merge': 'after_integration', 'post_segment': 'after_integration',
            'initial': 'initialization', 'after_integration': 'after_integration',
            'final': 'final', 'periodic': 'periodic'}[state]


def batch_target(task_ids):
    tasks = list(dict.fromkeys(task_ids))
    return tasks[0] if len(tasks) == 1 else tasks
