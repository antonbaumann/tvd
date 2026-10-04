"""Static task-vector integration, including the continuing AdamW state."""
from __future__ import annotations

import copy
import math
from typing import Any

import torch


def coefficient(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("lambda must be finite and between zero and one")
    return value


def clone_floating_state(model):
    return {name: tensor.detach().to(device="cpu", copy=True)
            for name, tensor in model.state_dict().items() if torch.is_floating_point(tensor)}


def apply_reptile_state(model, before, after, meta_lr):
    """Use source arithmetic at interior coefficients; exact copies at endpoints."""
    lam = coefficient(meta_lr)
    with torch.no_grad():
        state = model.state_dict()
        for name, old in before.items():
            target = state[name]
            if not target.is_floating_point():
                continue
            old = old.to(target.device, target.dtype)
            local = after[name].to(target.device, target.dtype)
            target.copy_(old if lam == 0 else local if lam == 1 else old + lam * (local - old))


def merge_model_state(model, before, meta_lr):
    apply_reptile_state(model, before, model.state_dict(), meta_lr)




def clone_optimizer_state(optimizer):
    return {param: {key: value.detach().to(device="cpu", copy=True) if torch.is_tensor(value)
                    else copy.deepcopy(value) for key, value in state.items()}
            for param, state in optimizer.state.items()}


def optimizer_tensor_like(value, reference):
    if value is None:
        return torch.zeros_like(reference)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def merge_state_value(state, before, key, meta_lr):
    if key not in state:
        return
    local = state[key]
    if torch.is_tensor(local):
        if key == "step" and not local.is_floating_point():
            # Old checkpoints can encode an integral counter; integration is fractional.
            local = state[key] = local.to(torch.float64)
        old = optimizer_tensor_like(before.get(key), local)
        local.copy_(old if meta_lr == 0 else old + meta_lr * (local - old))
    elif isinstance(local, (float, int)):
        old = float(before.get(key, 0))
        state[key] = old if meta_lr == 0 else old + meta_lr * (float(local) - old)


def merge_second_moment(state, before, key, meta_lr):
    if key not in state or not torch.is_tensor(state[key]):
        return
    local = state[key]
    old = optimizer_tensor_like(before.get(key), local)
    if meta_lr == 0:
        local.copy_(old)
    else:
        local.copy_(old.clamp_min(0).sqrt().lerp(local.clamp_min(0).sqrt(), meta_lr).square())


def merge_adamw_optimizer_state(optimizer, before, meta_lr):
    lam = coefficient(meta_lr)
    if lam == 1:
        return  # A sqrt/square round trip changes continuation, even at lambda=1.
    with torch.no_grad():
        for param, state in optimizer.state.items():
            old = before.get(param, {})
            for key in ("exp_avg", "step"):
                merge_state_value(state, old, key, lam)
            for key in ("exp_avg_sq", "max_exp_avg_sq"):
                merge_second_moment(state, old, key, lam)


merge_optimizer_state = merge_adamw_optimizer_state


def scheduled_meta_lr(base_meta_lr: float, local_steps: int, merge_config: dict[str, Any]) -> float:
    """Historical call interface; only the paper's static coefficients are supported."""
    if merge_config.get("mode", "static") != "static" or float(merge_config.get("alpha", 0)) != 0:
        raise ValueError("Paper trajectories require static lambda (mode=static, alpha=0)")
    if merge_config.get("max_meta_lr") is not None:
        raise ValueError("Coefficient schedules/caps are outside the paper implementation")
    return coefficient(base_meta_lr)
