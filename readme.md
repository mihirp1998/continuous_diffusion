mkdir tmp
cd tmp  
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

mv DeepSeek-OCR/tokenizer.json ../DeepSeek-OCR-code/
mv DeepSeek-OCR/model-00001-of-000001.safetensors ../DeepSeek-OCR-code/
cd ..
rm -rf tmp

---------------------------- install unsloth deepseek ocr ---------------------------- 
python download_unsloth_ocr.py 

pip install einops
pip install hydra-core
pip install addict
pip install matplotlib
pip install easydict

# set trace and doubel check if installs are causing bugs such as :

config file not found or model name not found