#
# Created on Tue Nov 15 2022 by Nguyen Qui Vinh Quang
#
# @licensed: Computer Vision & Image Processing Lab
# @project: VehicleReid
#


from .lr_scheduler import WarmupMultiStepLR, CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

import torch
from timm.scheduler import create_scheduler, cosine_lr
from src.utils.utils import config2object


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg["BASE_LR"]
        weight_decay = cfg["WEIGHT_DECAY"]
        if "bias" in key:
            lr = cfg["BASE_LR"] * cfg["BIAS_LR_FACTOR"]
            weight_decay = cfg["WEIGHT_DECAY_BIAS"]
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg["OPTIMIZER_NAME"] == "SGD":
        optimizer = getattr(torch.optim, cfg["OPTIMIZER_NAME"])(
            params, momentum=cfg["MOMENTUM"]
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    else:
        optimizer = getattr(torch.optim, cfg["OPTIMIZER_NAME"])(params)
    return optimizer


def make_scheduler(cfg, optimizer):
    if cfg["WARMUP_METHOD"] == "cosine":
        scheduler = __make_cosine_scheduler(cfg, optimizer)
    elif cfg["WARMUP_METHOD"] == "CosineAnnealing":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=200,
            cycle_mult=1.0,
            max_lr=0.01,
            min_lr=0.00001,
            warmup_steps=50,
            gamma=0.5,
        )
    elif cfg["WARMUP_METHOD"] == "StepLR":
        scheduler = StepLR(optimizer, step_size=cfg["STEPS"], gamma=0.5)
    else:
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg["STEPS"],
            cfg["GAMMA"],
            cfg["WARMUP_FACTOR"],
            cfg["WARMUP_ITERS"],
            cfg["WARMUP_METHOD"],
        )

    return scheduler


def __make_cosine_scheduler(cfg, optimizer):
    # Configuration detail:
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/scheduler/scheduler_factory.py
    _config = {
        "sched": "cosine",
        "warmup_epochs": 5,
        "lr_noise": None,
        "lr_noise_pct": 0.67,
        "lr_noise_std": 1,
        "warmup_lr": 1e-6,
        "min_lr": 1e-5,
        "cooldown_epochs": 10,
        "decay-rate": 0.1,
        "epochs": cfg["MAX_EPOCHS"],
        "lr_cycle_limit": 1,
        "seed": cfg["SEED"],
        "lr_cycle_mul": 1.0,
        "lr_cycle_decay": 0.1,
        "lr_cycle_limit": 1,
        "lr_k_decay": 1.0,
    }
    _config = config2object(_config)
    scheduler, epochs = create_scheduler(_config, optimizer=optimizer)
    print(scheduler)
    return scheduler
