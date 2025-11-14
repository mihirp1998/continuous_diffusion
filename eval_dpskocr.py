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
model, tokenizer = FastVisionModel.from_pretrained(
    # "deepseek_ocr",
    "outputs/robust-dust-7/checkpoint-60",
    load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
    auto_model = AutoModel,
    trust_remote_code=True,
    # unsloth_force_compile=True,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
# st()
FastVisionModel.for_inference(model) # Enable for inference!

prompt = "<image>\nFree OCR. "
image_file = '/home/mprabhud/phd_projects/continuous_diffusion/story_00000001.png'
# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = model.infer(tokenizer, prompt=prompt, image_file=image_file,
    output_path = 'out/',
    image_size=640,
    base_size=1024,
    crop_mode=True,
    eval_mode = True,
    save_results = False,
    test_compress = False)
print(res)