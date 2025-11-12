from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer
import ipdb
st = ipdb.set_trace
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'
model_name = "DeepSeek-OCR-code"  # Commented out to use HuggingFace model
# model_name = "tmp/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
# base_size = 512, image_size = 512, crop_mode = False

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'story_00000001.png'
output_path = 'out/'
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
    image_tensor = image_tensor.repeat(2, 1, 1, 1)  # Shape: [2, C, H, W]
else:
    image_tensor = None

# st()

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
res = model.get_image_features(image_tensor.cuda())
st()
# res = model.get_image_features(tokenizer, prompt=prompt, image_tensor=image_tensor, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
res[0] = res[0] + torch.randn_like(res[0]) *0.1
st()
# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
res_text =  model.infer(tokenizer,image_features=[res[:1]], prompt=prompt, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
# res =model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, eval_mode = True)
print(res_text)
st()

