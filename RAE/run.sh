

torchrun --standalone --nproc_per_node=8 src/train_stage1.py --config configs/stage1/training/DINOv2-B_decXL.yaml --data-path /data/user_data/mprabhud/imagenet-mini --results-dir results/stage1 --image-size 256 --precision bf16

# stage 2 training

torchrun --standalone --nnodes=1 --nproc_per_node=8 src/train.py --config /home/mprabhud/phd_projects/continuous_diffusion/RAE/configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B.yaml  --data-path /data/user_data/mprabhud/imagenet-mini  --results-dir results/stage2  --precision bf16

torchrun --standalone --nnodes=1 --nproc_per_node=1 src/train.py exps="[tiny,ocr]"