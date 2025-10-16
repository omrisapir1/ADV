import torch
from torch.optim import AdamW
from typing import Dict, Any


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    optim_config = config.get("optim", {})

    # Get parameters that should not have weight decay
    no_decay_modules = optim_config.get("no_decay_modules", ["LayerNorm", "bias"])

    # Separate parameters into decay and no_decay groups
    decay_params, no_decay_params = [], []
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        should_not_decay = any(module_name in name for module_name in no_decay_modules)
        (no_decay_params if should_not_decay else decay_params).append(param)

    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": optim_config.get("weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Create optimizer
    if optim_config.get("name", "adamw").lower() == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=float(optim_config.get("lr")),
            betas=optim_config.get("betas", (0.9, 0.999)),
            eps=float(optim_config.get("eps", 1e-8)),
            fused=optim_config.get("fused", False),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_config.get('name')}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], total_steps: int):
    """Return a constant (no-decay, no-warmup) scheduler."""
    # ConstantLR with factor=1.0 keeps LR unchanged.
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
