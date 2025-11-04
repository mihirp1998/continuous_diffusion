from transformers import AutoModel, AutoTokenizer
import ipdb
st = ipdb.set_trace
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# model_name = 'deepseek-ai/DeepSeek-OCR'
model_name = "DeepSeek-OCR-code"  # Commented out to use HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)
# base_size = 512, image_size = 512, crop_mode = False

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = '/home/mprabhud/datasets/tinystories/image_dataset/story_00000001.png'
output_path = 'out/'
# st()

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
res = model.get_image_features(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True, return_image_features = True)
# res[0] = res[0] +0.1

# res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
res_text =  model.infer(tokenizer,image_features=res, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True)
# res =model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, eval_mode = True)
# st()
print(res)
