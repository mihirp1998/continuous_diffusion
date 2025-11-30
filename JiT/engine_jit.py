import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
import ipdb
st = ipdb.set_trace

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None, encoder=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32)
        # st()
        
        
        if args.use_features and encoder is not None and args.txt_modeling and args.dataset == "tiny":
            # assumes x to be in [0, 1]
            with torch.no_grad():
                x = encoder.get_image_features(x)
                bs, num_tokens, dim = x.shape
                x = x.permute(0, 2, 1)
                h = w = int(math.sqrt(num_tokens))
                x = x.view(bs, dim, h, w)                
        else:
            x = x.div_(255)
            x = x * 2.0 - 1.0
        
        if args.txt_modeling:
            labels = None
        else:
            labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in wandb to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.log({'train_loss': loss_value_reduce, 'lr': lr, 'step': epoch_1000x})
    return x


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None, device=None, encoder=None, encoder_tokenizer=None, input_images=None, eval_model=None, eval_tokenizer=None):
    # st()
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    if args.eval_ema:
        model_without_ddp.load_state_dict(ema_state_dict)
    
    # st()

    # ensure that the number of images per class is equal.
    if args.txt_modeling:
        class_label_gen_world = None
    else:        
        class_num = args.class_num
        assert args.num_images % class_num == 0, "Number of images per class must be the same"
        class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
        class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    # st()
    all_perplexities = []
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        if class_label_gen_world is not None:
            labels_gen = class_label_gen_world[start_idx:end_idx]
            labels_gen = torch.Tensor(labels_gen).long().cuda()
        else:
            labels_gen = None

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen, batch_size=batch_size, device= device)
        # st()
        torch.distributed.barrier()
        if args.txt_modeling:
            prompt = "<image>\nFree OCR. "
            
            sampled_images = sampled_images.flatten(2).permute(0, 2, 1).to(torch.bfloat16)
            # sampled_images = input_images.flatten(2).permute(0, 2, 1).to(torch.bfloat16)
            out_text = encoder.infer(encoder_tokenizer,image_features=sampled_images.unsqueeze(1), prompt=prompt, base_size = 512, image_size = 512, crop_mode = False, eval_mode = True,  max_new_tokens=256)
            # out_text = [''] * len(out_text)
            
            out_text = [text_val for text_val in out_text if text_val != '']
            
            
            if eval_model is not None:
                if len(out_text) > 0:
                    eval_inputs = eval_tokenizer(out_text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
                    with torch.no_grad():
                        eval_outputs = eval_model(**eval_inputs, labels=eval_inputs.input_ids)
                    loss = eval_outputs.loss
                    perplexity = torch.exp(loss).item()
                    all_perplexities.append(perplexity)
                else:
                    all_perplexities.append(1000000)
            
            
            # st()
        else:
            # denormalize images
            sampled_images = (sampled_images + 1) / 2
            sampled_images = sampled_images.detach().cpu()

        # distributed save images
        if not args.txt_modeling:
            for b_id in range(sampled_images.size(0)):
                img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= args.num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)
    
    torch.distributed.barrier()
    # st()
    # print("checking 2")
    # aggregate perplexities across all processes
    if args.txt_modeling and eval_model is not None:
        print(f"[Rank {misc.get_rank()}] checking perplexity")
        # Ensure all processes participate in the all_gather, even if they have no valid perplexities
        if len(all_perplexities) > 0 and not np.isnan(np.mean(all_perplexities)):
            local_mean = torch.tensor([np.mean(all_perplexities)], device=device).to(torch.float32)
            local_count = torch.tensor([1.0], device=device)
        else:
            local_mean = torch.tensor([0.0], device=device).to(torch.float32)
            local_count = torch.tensor([0.0], device=device)
        
        all_means = [torch.zeros(1, device=device) for _ in range(world_size)]
        all_counts = [torch.zeros(1, device=device) for _ in range(world_size)]
        torch.distributed.all_gather(all_means, local_mean)
        torch.distributed.all_gather(all_counts, local_count)
        
        if misc.get_rank() == 0:
            total_count = torch.stack(all_counts).sum().item()
            if total_count > 0:
                global_mean = (torch.stack(all_means) * torch.stack(all_counts)).sum().item() / total_count
                print("Average perplexity (all processes): ", global_mean)
                
                if log_writer is not None:
                    log_writer.log({'perplexity_mean': global_mean, 'epoch': epoch})
        
        if len(all_perplexities) > 0:
            print(f"[Rank {misc.get_rank()}] Average perplexity: ", np.mean(all_perplexities), "Std: ", np.std(all_perplexities))
        

    # back to no ema
    print(f"[Rank {misc.get_rank()}] Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    print(f"[Rank {misc.get_rank()}] loaded model state dict")
    
    if not args.txt_modeling:
        # compute FID and IS
        if log_writer is not None:
            if args.img_size == 256:
                fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
            elif args.img_size == 512:
                fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
            else:
                raise NotImplementedError
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=save_folder,
                input2=None,
                fid_statistics_file=fid_statistics_file,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            fid = metrics_dict['frechet_inception_distance']
            inception_score = metrics_dict['inception_score_mean']
            postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
            log_writer.log({'fid{}'.format(postfix): fid, 'is{}'.format(postfix): inception_score, 'epoch': epoch})
            print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
            shutil.rmtree(save_folder)
    print(f"[Rank {misc.get_rank()}] before final barrier")
    torch.distributed.barrier()
    print(f"[Rank {misc.get_rank()}] after final barrier")