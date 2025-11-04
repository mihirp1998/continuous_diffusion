import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
from omegaconf import DictConfig, OmegaConf
import ipdb
st = ipdb.set_trace

def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    # Handle DictConfig from Hydra/OmegaConf - convert to plain dict first
    if isinstance(namespace, DictConfig):
        namespace = OmegaConf.to_container(namespace, resolve=True)
    
    # Handle regular dicts - recursively convert nested structures
    if isinstance(namespace, dict):
        result = {}
        for k, v in namespace.items():
            # Ensure keys are JSON-serializable (convert tuples to strings if needed)
            if isinstance(k, tuple):
                k = str(k)
            # Recursively process nested dicts, Namespaces, and DictConfigs
            if isinstance(v, (dict, DictConfig, argparse.Namespace)):
                result[k] = namespace_to_dict(v)
            else:
                result[k] = v
        return result
    
    # Handle argparse.Namespace
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    wandb.login()
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(0,1))
    x = x.clamp(0, 1).mul(255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x