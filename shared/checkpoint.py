"""Continuing checkpoints at completed integration boundaries."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from shared.io import JsonlLogger, sha256


def stream_path(config):
    return Path(config["stream"])


def rng_state(*, cuda_device=None):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": []}
    if torch.cuda.is_available():
        if cuda_device is None:
            state['cuda'] = torch.cuda.get_rng_state_all()
        else:
            state.update(cuda=[torch.cuda.get_rng_state(cuda_device)], cuda_device=cuda_device)
    return state


def restore_rng(state, *, cuda_device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        if 'cuda_device' in state:
            torch.cuda.set_rng_state(state['cuda'][0].cpu(), device=state['cuda_device'] if cuda_device is None else cuda_device)
        elif cuda_device is not None:
            # Older checkpoints saved all visible devices on every rank.
            torch.cuda.set_rng_state(state['cuda'][cuda_device].cpu(), device=cuda_device)
        else:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def read_checkpoint(config):
    path = config.get("_runtime", {}).get("resume")
    if not path:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "tvd-boundary-v1":
        raise ValueError("Resume requires a consolidated continuing-boundary checkpoint")
    stream = stream_path(config) if config.get("stream") else None
    if stream and checkpoint.get("stream_sha256") != sha256(stream):
        raise ValueError("Resume stream differs from the checkpoint's exact stream")
    world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    if checkpoint["world_size"] != world:
        raise ValueError("Exact boundary resume requires the original world size")
    old = checkpoint["config"]
    if not config.get("_runtime", {}).get("eval_only"):
        ignored = {"max_steps", "max_segments", "checkpoint_every_segments", "save_model", "final_model_path"}
        for key in ("seed", "model"):
            if config.get(key) != old.get(key):
                raise ValueError(f"Resume changes {key}; use the checkpoint's original configuration")
        current_train = {k: v for k, v in config.get("train", {}).items() if k not in ignored}
        previous_train = {k: v for k, v in old.get("train", {}).items() if k not in ignored}
        if current_train != previous_train:
            raise ValueError("Resume changes training hyperparameters")
    return checkpoint


def restore(checkpoint, model, optimizer=None, scheduler=None, *, cuda_device=None):
    if checkpoint is None:
        return {"step": 0, "sequence": 0, "extra": {}}
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    restore_rng(checkpoint["rng"][rank], cuda_device=cuda_device)
    return checkpoint


def save(config, model, optimizer, step, sequence, *, scheduler=None, extra=None, final=False, cuda_device=None):
    every = int(config.get("train", {}).get("checkpoint_every_segments", 0))
    if not final and (not every or sequence % every):
        return
    distributed = torch.distributed.is_initialized()
    world = torch.distributed.get_world_size() if distributed else 1
    rank = torch.distributed.get_rank() if distributed else 0
    states = [None] * world
    if distributed:
        torch.distributed.all_gather_object(states, rng_state(cuda_device=cuda_device))
    else:
        states[0] = rng_state(cuda_device=cuda_device)
    if rank == 0:
        directory = Path(config["output_dir"]) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / ("continuing.pt" if final else f"sequence-{sequence}.pt")
        temporary = destination.with_suffix(".tmp")
        stream = stream_path(config) if config.get("stream") else None
        torch.save({"format": "tvd-boundary-v1", "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "step": step, "sequence": sequence, "extra": extra or {}, "rng": states,
                    "world_size": world, "stream_sha256": sha256(stream) if stream else None,
                    "config": config}, temporary)
        os.replace(temporary, destination)
        logger = JsonlLogger(directory.parent / "metadata.jsonl", append=True)
        logger.write({"event": "checkpoint", "step": step, "sequence": sequence,
                      "path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size})
        logger.close()
    if distributed:
        torch.distributed.barrier()
