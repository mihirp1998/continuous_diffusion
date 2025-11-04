from datasets import load_dataset
import cv2
import numpy as np
import textwrap
from tqdm import tqdm
from RAE.src.utils.basic_utils import create_text_image
import ipdb
st = ipdb.set_trace
import os

root_dir = "/data/user_data/mprabhud/tiny_story_dataset"
root_dir  = "/home/mprabhud/datasets/tinystories"
# Load TinyStories dataset
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

# Create output directory
# os.makedirs("tinystories_images", exist_ok=True)

os.makedirs(f"{root_dir}/image_dataset", exist_ok=True)
os.makedirs(f"{root_dir}/text_dataset", exist_ok=True)

# Function to create image from text
# st()

from transformers import AutoModel, AutoTokenizer
import ipdb
st = ipdb.set_trace
import torch
import os
# model_name = 'deepseek-ai/DeepSeek-OCR'
model_name = "DeepSeek-OCR-code"  # Commented out to use HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = '/home/mprabhud/datasets/tinystories/image_dataset/story_00000001.png'
output_path = 'out/'
root_dir = "/home/mprabhud//tinyimages"

# Generate images from first 10 stories
print("Creating images from TinyStories...")
import sys

start_index = int(sys.argv[1])  # Set your desired starting index here

start_actual_index = start_index * 250000

for i, story in enumerate(tqdm(dataset)):
    if i < start_actual_index:
        continue
    text = story['text']
    img = create_text_image(text)
    
    
    
    if img is not None:
        # Save image
        os.makedirs(f"{root_dir}/image_dataset", exist_ok=True)
        filename = f"{root_dir}/image_dataset/story_{i+1:08d}.png"
        cv2.imwrite(filename, img)
        # st()
        res = model.get_image_features(tokenizer, prompt=prompt, image_file=filename, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
        os.makedirs(f"{root_dir}/image_features", exist_ok=True)
        torch.save(res, f"{root_dir}/image_features/story_{i+1:08d}.pt")
        os.makedirs(f"{root_dir}/text_dataset", exist_ok=True)
        with open(f"{root_dir}/text_dataset/story_{i+1:08d}.txt", "w") as f:
            f.write(text)
        # print(f"Saved: {filename}")
        # st()

# print("Done! Images saved in 'tinystories_images' directory")
