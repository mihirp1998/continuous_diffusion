import argparse
import datetime
import numpy as np
import os
import time
import ipdb
st = ipdb.set_trace
from pathlib import Path
import sys
sys.path.append("../RAE/src")
from utils.basic_utils import create_text_image, TextDataset
from utils.model_utils import load_model_from_ddp_checkpoint
from unsloth import FastVisionModel
import torch
import torch.backends.cudnn as cudnn
import wandb
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from util.crop import center_crop_arr
import util.misc as misc

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # Convert OmegaConf to argparse-like object for compatibility
    args = OmegaConf.to_object(cfg)
    args = argparse.Namespace(**args)
    
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    # st()

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up wandb logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(
            project="jit-training",
            config=vars(args),
            dir=args.output_dir
        )
        log_writer = wandb
    else:
        log_writer = None
    
    # Data augmentation transforms
    if cfg.dataset == "imagenet":
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor()
        ])

        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'imagenet', 'train'), transform=transform_train)
    else:
        transform = transforms.Compose([
            # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])        
        dataset_train = TextDataset(cfg.data_path, transform=transform, num_stories=cfg.training.num_examples)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    if args.encoder == "ocr-noise":
        print("Loading OCR noise encoder")
        ckpt_dir = os.environ['CKPT_DIR']
        # st()
        if args.ocr_path is None:
            ocr_path = None
        else:
            ocr_path = f"{ckpt_dir}/{args.ocr_path}"
        encoder_model, encoder_tokenizer = load_model_from_ddp_checkpoint(ocr_path)
        encoder_model.to(device)
        FastVisionModel.for_inference(encoder_model)         
        encoder = encoder_model
        print("OCR noise encoder loaded")
    else:
        encoder = None   
        encoder_tokenizer = None

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    # st()

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)
    
    if args.do_gen_perplexity:
        print("Loading GPT-2 model for perplexity evaluation")
        eval_model = GPT2LMHeadModel.from_pretrained("gpt2")
        eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
        eval_model.eval().to(device)


    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # st()
    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, args, 0, batch_size=args.gen_bsz, log_writer=log_writer, device=device, encoder=encoder, encoder_tokenizer=encoder_tokenizer, eval_model=eval_model, eval_tokenizer=eval_tokenizer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # st()
        x = train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args, encoder=encoder)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer, device=device, encoder=encoder, encoder_tokenizer=encoder_tokenizer, input_images=x)
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    main()
