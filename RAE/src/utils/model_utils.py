import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import RAE
import torch.nn as nn
from omegaconf import OmegaConf
import torch
import ipdb
st = ipdb.set_trace
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from transformers import AutoModel, AutoTokenizer

def load_model_from_ddp_checkpoint(checkpoint_path):
    """Load model from DDP checkpoint by removing 'module.' prefix from state dict keys"""
    import os
    from collections import OrderedDict
    os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
    model, tokenizer = FastVisionModel.from_pretrained(
        "/home/mprabhud/phd_projects/continuous_diffusion//deepseek_ocr",  # Load base model first
        load_in_4bit = False,
        auto_model = AutoModel,
        trust_remote_code=True,
        random_noise=0.5,
        use_gradient_checkpointing = "unsloth",
    )    
    # st()
    # Load the checkpoint state dict
    # Load the checkpoint using the safetensors files
    if checkpoint_path is not None:
        from safetensors.torch import load_file
        print("reading checkpoint shards")
        # Load model shards
        shard_1 = load_file(f"{checkpoint_path}/model-00001-of-00002.safetensors")
        shard_2 = load_file(f"{checkpoint_path}/model-00002-of-00002.safetensors")
        print("loaded checkpoint shards")
        # Combine shards
        checkpoint = {**shard_1, **shard_2}
        
        # Remove 'module.' prefix from DDP wrapped model keys
        new_state_dict = OrderedDict()
        for key, value in checkpoint.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load the cleaned state dict
        model.load_state_dict(new_state_dict, strict=True)
    print("checkpoint loaded successfully")
    
    return model, tokenizer
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if config["target"] == "ocr-noise":
        model, tokenizer = load_model_from_ddp_checkpoint("/grogu/user/mprabhud/dpsk_ckpts/quiet-grass-102/checkpoint-2000")
        # model, tokenizer = load_model_from_ddp_checkpoint(None)
        FastVisionModel.for_inference(model) 
    elif config["target"] == "ocr":
        import torch
        model_name = "/home/mprabhud/phd_projects/continuous_diffusion/DeepSeek-OCR-code"  # Commented out to use HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
        model = model.eval().cuda().to(torch.bfloat16)        
    else:
        # st()
        model = get_obj_from_str(config["target"])(**config.get("params", dict()))
        ckpt_path = config.get("ckpt", None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # see if it's a ckpt from training by checking for "model"
            if "ema" in state_dict:
                state_dict = state_dict["ema"]
            elif "model" in state_dict:
                raise NotImplementedError("Loading from 'model' key not implemented yet.")
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=True)
            print(f'target {config["target"]} loaded from {ckpt_path}')
    return model

