mkdir tmp
cd tmp  
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

mv DeepSeek-OCR/tokenizer.json ../DeepSeek-OCR-code/
mv DeepSeek-OCR/model-00001-of-000001.safetensors ../DeepSeek-OCR-code/
cd ..
rm -rf tmp