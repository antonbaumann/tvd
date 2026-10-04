"""Record the exact runtime inputs supplied by each experiment."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from shared.io import RUN_CONTEXT, JsonlLogger, sha256


def start_run(config):
    """Attach explicit run metadata and installed dependency versions to JSONL events."""
    RUN_CONTEXT.clear()
    RUN_CONTEXT.update(config.pop('_tvd'))
    versions = {}
    for name in ('torch', 'numpy', 'transformers', 'accelerate', 'datasets', 'trl', 'python-dotenv', 'filelock', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    RUN_CONTEXT['dependency_versions'] = versions


def tokenizer_metadata(tokenizer):
    return {
        "name": tokenizer.name_or_path,
        # Locally constructed tokenizers have no Hub commit.
        "revision": tokenizer.init_kwargs.get("_commit_hash"),
        "chat_template": tokenizer.chat_template,
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
        ).hexdigest(),
    }


def record_runtime(*, output_dir: Path, model_config: dict, model_revision: str | None, tokenizer: dict, input_files: list[Path]):
    files = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
             for path in input_files]
    logger = JsonlLogger(output_dir / "metadata.jsonl", append=True)
    logger.write({"event": "runtime_provenance", "model_config": model_config, "model_revision": model_revision,
                  "tokenizer": tokenizer, "input_files": files})
    logger.close()
