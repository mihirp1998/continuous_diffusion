torchrun --nproc_per_node=8 --nnodes=1 main_jit.py exps=ocr_noise

# torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
# main_jit.py \
# --model JiT-B/16 \
# --proj_dropout 0.0 \
# --P_mean -0.8 --P_std 0.8 \
# --img_size 256 --noise_scale 1.0 \
# --batch_size 128 --blr 5e-5 \
# --epochs 600 --warmup_epochs 5 \
# --gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
# --output_dir /grogu/user/mprabhud/jit_ckpt --resume /grogu/user/mprabhud/jit_ckpt \
# --data_path /grogu/datasets/imagenet --online_eval

# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
# main_jit.py \
# --model JiT-B/16 \
# --proj_dropout 0.0 \
# --P_mean -0.8 --P_std 0.8 \
# --img_size 256 --noise_scale 1.0 \
# --batch_size 128 --blr 5e-5 \
# --epochs 600 --warmup_epochs 5 \
# --gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
# --output_dir /grogu/user/mprabhud/jit_ckpt --resume /grogu/user/mprabhud/jit_ckpt \
# --data_path /grogu/datasets/imagenet --online_eval


torchrun --nproc_per_node=1 --nnodes=1 main_jit.py exps=ocr_noise resume=../../jit_ckpt/ evaluate_gen=True do_gen_perplexity=True num_sampling_steps=100

torchrun --nproc_per_node=1 --nnodes=1 main_jit.py exps=ocr_noise model="JiT-E-nb" online_eval=True