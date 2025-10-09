import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, Any, Iterator


def create_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    optim_config = config.get("optim", {})

    # Get parameters that should not have weight decay
    no_decay_modules = optim_config.get("no_decay_modules", ["LayerNorm", "bias"])

    # Separate parameters into decay and no_decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this parameter should have no weight decay
        should_not_decay = any(module_name in name for module_name in no_decay_modules)

        if should_not_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": optim_config.get("weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    # Create optimizer
    if optim_config.get("name", "adamw").lower() == "adamw":
        optimizer = AdamW(
            param_groups,
            lr=optim_config.get("lr"),
            betas=optim_config.get("betas"),
            eps=optim_config.get("eps"),
            fused=optim_config.get("fused")
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_config.get('name')}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any], total_steps: int):
    """Create learning rate scheduler based on config."""
    scheduler_config = config.get("scheduler", {})

    if scheduler_config.get("name", "linear").lower() == "linear":
        warmup_ratio = scheduler_config.get("warmup_ratio", 0.03)
        warmup_steps = int(total_steps * warmup_ratio)

        scheduler = LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        return scheduler
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config.get('name')}")

