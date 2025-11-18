from unsloth import FastVisionModel # FastLanguageModel for LLMs
import ipdb
st = ipdb.set_trace
import torch
import wandb
from transformers import AutoModel
import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'
# # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# fourbit_models = [
#     "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit", # Qwen 3 vision support
#     "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
#     "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
#     "unsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
# ] # More models at https://huggingface.co/unsloth
# from huggingface_hub import snapshot_download
# snapshot_download("unsloth/DeepSeek-OCR", local_dir = "deepseek_ocr")
# st()
# Load the model and handle DDP state dict keys
import torch
from collections import OrderedDict
import ipdb; 
st = ipdb.set_trace

def load_model_from_ddp_checkpoint(checkpoint_path):
    """Load model from DDP checkpoint by removing 'module.' prefix from state dict keys"""
    model, tokenizer = FastVisionModel.from_pretrained(
        "deepseek_ocr",  # Load base model first
        load_in_4bit = False,
        auto_model = AutoModel,
        trust_remote_code=True,
        random_noise=0.0,
        use_gradient_checkpointing = "unsloth",
    )    
    # Load the checkpoint state dict
    # Load the checkpoint using the safetensors files
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

# model, tokenizer = load_model_from_ddp_checkpoint("/grogu/user/mprabhud/dpsk_ckpts/pleasant-sound-101/checkpoint-3750")
model, tokenizer = load_model_from_ddp_checkpoint("/grogu/user/mprabhud/dpsk_ckpts/quiet-grass-102/checkpoint-2000")

# st()
FastVisionModel.for_inference(model) # Enable for inference!

prompt = "<image>\nFree OCR. "
image_file = '/home/mprabhud/phd_projects/continuous_diffusion/story_00000001.png'
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
print("inferring")
res = model.infer(tokenizer, prompt=prompt, image_file=image_file,
    output_path = 'out/',
    image_size=512,
    base_size=512,    
    crop_mode=True,
    eval_mode = True,
    max_new_tokens=256,
    save_results = False,
    test_compress = False)
print(res)