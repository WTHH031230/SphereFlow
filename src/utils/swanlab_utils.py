import swanlab
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
import os
import argparse
import math

def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def initialize(args, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    if is_main_process():
        swanlab.init(
            project=project_name,
            experiment_name=exp_name,
            config=config_dict,
        )

def log(stats, step=None):
    if is_main_process():
        swanlab.log({k: v for k, v in stats.items()}, step=step)

def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        swanlab.log({"samples": swanlab.Image(sample)}, step=step)

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
