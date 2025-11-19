import torch
from torch.optim import AdamW
from typing import Dict, Any


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on config (no default values in .get)."""
    optim_config = config.get("optim")


    no_decay_modules = optim_config.get("no_decay_modules")

    decay_params, no_decay_params = [], []
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        should_not_decay = any(module_name in name for module_name in (no_decay_modules or []))
        (no_decay_params if should_not_decay else decay_params).append(param)

    weight_decay_val = optim_config.get("weight_decay")
    param_groups = [
        {"params": decay_params, "weight_decay": 0.0},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    name_val = optim_config.get("name")
    if name_val.lower() == "adamw":
        lr_val = optim_config.get("lr")
        betas_val = optim_config.get("betas")
        eps_val = optim_config.get("eps")
        optimizer = AdamW(
            param_groups,
            lr=float(lr_val),
            betas=betas_val,
            eps=float(eps_val),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name_val}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, config: Dict[str, Any] = None):
    """Return a constant scheduler (no defaults)."""
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    if not config:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    optim_config = config.get("optim")
    warmup = optim_config.get("warmup")
    warmup_steps = warmup.get("warmup_steps")
    warmup_lr = warmup.get("lr")
    base_lr = optim_config.get("lr")

    def lr_lambda(step):
        if step < warmup_steps:
            # LR should be 1e-6
            return float(warmup_lr) / float(base_lr)       # scales base_lr to warmup_lr
        else:
            # LR should be 1e-5
            return 1.0                       # base LR stays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

