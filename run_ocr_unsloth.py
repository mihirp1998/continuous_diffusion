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



# # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# # model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
# model = model.eval().cuda().to(torch.bfloat16)
# base_size = 512, image_size = 512, crop_mode = False

model, tokenizer = load_model_from_ddp_checkpoint("/grogu/user/mprabhud/dpsk_ckpts/quiet-grass-102/checkpoint-2000")
# model, tokenizer = load_model_from_ddp_checkpoint(None)
# st()
FastVisionModel.for_inference(model) # Enable for inference!

prompt = "<image>\nFree OCR. "
# image_file = '/home/mprabhud/phd_projects/continuous_diffusion/story_00000001.png'
image_file = "/home/mprabhud/datasets/tiny_rawdata/00000.png"


# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# image_file = 'story_00000001.png'
# output_path = 'out/'
use_image_tensor = True

if use_image_tensor:
    from PIL import Image
    import torchvision.transforms as transforms

    # Load image using PIL
    pil_image = Image.open(image_file).convert('RGB')

    # Convert PIL image to tensor with values from 0 to 1
    transform = transforms.ToTensor()
    image_tensor = transform(pil_image)  # Shape: [C, H, W], values in [0, 1]

    # Add batch dimension and repeat to create batch of size 2
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    # image_tensor = image_tensor.repeat(0, 1, 1, 1)  # Shape: [2, C, H, W]
else:
    image_tensor = None

# st()
load_second_image = True
if load_second_image:
    image_file = "/home/mprabhud/datasets/tiny_rawdata/00001.png"
    pil_image = Image.open(image_file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor_2 = transform(pil_image)  # Shape: [C, H, W], values in [0, 1]
    image_tensor_2 = image_tensor_2.unsqueeze(0)  # Shape: [1, C, H, W]
    # image_tensor = image_tensor.repeat(0, 1, 1, 1)  # Shape: [2, C, H, W]
    image_tensor = torch.cat([image_tensor, image_tensor_2], dim=0)

# st()
# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
res = model.get_image_features(image_tensor.cuda())
# st()
# res = model.get_image_features(tokenizer, prompt=prompt, image_tensor=image_tensor, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
# res[0] = res[0] + torch.randn_like(res[0]) *0.5
# st()
image_features_val = [res[:1]]
image_features_val = res.unsqueeze(1)
# st()
# image_features_val = res.repea
res_text =  model.infer(tokenizer,image_features=image_features_val, prompt=prompt, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)

st()
# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
# res_text =  model.infer(tokenizer,image_features=[res[:1]], prompt=prompt, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
# res =model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, eval_mode = True)

# res_text =  model.generate_text([res[:1]], tokenizer)
print(res_text)
# res_text = "Hello, world!"
# # st()
# compute_loss =  model.compute_loss([res[:1]], tokenizer, res_text)
# st()

