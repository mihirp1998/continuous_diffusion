from unsloth import FastVisionModel # FastLanguageModel for LLMs
import ipdb
st = ipdb.set_trace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from transformers import AutoModel
import os
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

# Initialize distributed training
def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize the process group
        dist.init_process_group(backend='nccl')
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU training
        return 0, 1, 0

# Only initialize distributed if running in distributed mode
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank, world_size, local_rank = setup_distributed()
else:
    rank, world_size, local_rank = 0, 1, 0

is_main_process = rank == 0

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Qwen3-VL-8B-Instruct-bnb-4bit", # Qwen 3 vision support
    "unsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
    "unsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
] # More models at https://huggingface.co/unsloth
from huggingface_hub import snapshot_download
# snapshot_download("unsloth/DeepSeek-OCR", local_dir = "deepseek_ocr")
# st()

# Only load model on main process first to avoid conflicts
if is_main_process:
    model, tokenizer = FastVisionModel.from_pretrained(
        "deepseek_ocr",
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        auto_model = AutoModel,
        trust_remote_code=True,
        random_noise = 0.8,        
        # unsloth_force_compile=True,
        # use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        use_gradient_checkpointing=False
    )
    
    # Synchronize all processes before others load the model
    if world_size > 1:
        dist.barrier()
else:
    # Wait for main process to finish loading
    if world_size > 1:
        dist.barrier()
    
    # Load model on other processes
    model, tokenizer = FastVisionModel.from_pretrained(
        "deepseek_ocr",
        load_in_4bit = False,
        auto_model = AutoModel,
        trust_remote_code=True,
        random_noise = 0.8,
        # use_gradient_checkpointing = "unsloth",
        use_gradient_checkpointing=False
    )

# st()
# @title Create datacollator

import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io

from deepseek_ocr.modeling_deepseekocr import (
    format_messages,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)

@dataclass
class DeepSeekOCRDataCollator:
    """
    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 640)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Get BOS token ID from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (bytes dict or PIL Image) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # For smaller base sizes, resize; for larger, pad
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
            """
            Process a single conversation into model inputs.
            """

            # --- 1. Setup ---
            images = []
            for message in messages:
                if "images" in message and message["images"]:
                    for img_data in message["images"]:
                        if img_data is not None:
                            pil_image = self.deserialize_image(img_data)
                            images.append(pil_image)

            if not images:
                raise ValueError("No images found in sample. Please ensure all samples contain images.")

            tokenized_str = []
            images_seq_mask = []
            images_list, images_crop_list, images_spatial_crop = [], [], []

            prompt_token_count = -1 # Index to start training
            assistant_started = False
            image_idx = 0

            # Add BOS token at the very beginning
            tokenized_str.append(self.bos_id)
            images_seq_mask.append(False)

            for message in messages:
                role = message["role"]
                content = message["content"]

                # Check if this is the assistant's turn
                if role == "<|Assistant|>":
                    if not assistant_started:
                        # This is the split point. All tokens added *so far*
                        # are part of the prompt.
                        prompt_token_count = len(tokenized_str)
                        assistant_started = True

                    # Append the EOS token string to the *end* of assistant content
                    content = f"{content.strip()} {self.tokenizer.eos_token}"

                # Split this message's content by the image token
                text_splits = content.split('<image>')

                for i, text_sep in enumerate(text_splits):
                    # Tokenize the text part
                    tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                    tokenized_str.extend(tokenized_sep)
                    images_seq_mask.extend([False] * len(tokenized_sep))

                    # If this text is followed by an <image> tag
                    if i < len(text_splits) - 1:
                        if image_idx >= len(images):
                            raise ValueError(
                                f"Data mismatch: Found '<image>' token but no corresponding image."
                            )

                        # Process the image
                        image = images[image_idx]
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        # Add image placeholder tokens
                        tokenized_str.extend(tok_img)
                        images_seq_mask.extend([True] * len(tok_img))

                        image_idx += 1 # Move to the next image

            # --- 3. Validation and Final Prep ---
            if image_idx != len(images):
                raise ValueError(
                    f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
                )

            # If we never found an assistant message, we're in a weird state
            # (e.g., user-only prompt). We mask everything.
            if not assistant_started:
                print("Warning: No assistant message found in sample. Masking all tokens.")
                prompt_token_count = len(tokenized_str)

            # Prepare image tensors
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

            return {
                "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
                "images_ori": images_ori,
                "images_crop": images_crop,
                "images_spatial_crop": images_spatial_crop_tensor,
                "prompt_token_count": prompt_token_count, # This is now accurate
            }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        batch_data = []

        # Process each sample
        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only=True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Prepare images batch (list of tuples)
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        # Stack spatial crop info
        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

from datasets import load_dataset
# dataset = load_dataset("hezarai/parsynth-ocr-200k", split = "train[:2000]")
use_peft = False

if use_peft:
    model = FastVisionModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
else:
    model.lm_head.requires_grad_(True)
    model.model.layers.requires_grad_(True)
    model.model.embed_tokens.requires_grad_(True)
    # model.model.sam_model.requires_grad_(True)
    # model.model.vision_model.requires_grad_(True)
    # model.model.projector.requires_grad_(True)
    model.model.norm.requires_grad_(True)
    # model.model.image_newline.requires_grad_(True)
    # model.model.view_seperator.requires_grad_(True)    
# Check number of trainable parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    return total_params, trainable_params, trainable_param_names



if is_main_process:
    total_params, trainable_params, trainable_param_names = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    non_trainable_params = total_params - trainable_params
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")

# Set requires_grad=True for specific model components

# st()

instruction = "<image>\nFree OCR. "

def convert_to_conversation(sample):
    """Convert dataset sample to conversation format"""
    # Handle both tuple (image, text) from TextDataset and dict from HuggingFace dataset
    if isinstance(sample, tuple):
        image, text = sample
    else:
        image = sample['image']
        text = sample['text']
    
    conversation = [
        {
            "role": "<|User|>",
            "content": instruction,
            "images": [image]
        },
        {
            "role": "<|Assistant|>",
            "content": text
        },
    ]
    return {"messages": conversation}

# Load dataset
# dataset = load_dataset("hezarai/parsynth-ocr-200k", split = "train[:1000]")
# dataset = dataset.rename_column("image_path", "image")
# Load dataset using TextDataset
from RAE.src.utils.basic_utils import TextDataset
train_dataset = TextDataset(root="./", num_stories=500000, convert_to_conversation=True)
val_dataset = TextDataset(root="./", num_stories=100, eval_mode=True, convert_to_conversation=True)  # Smaller validation set

if is_main_process:
    print("Converting train dataset to conversation format")
import time
start_time = time.time()
# Check if converted dataset exists, if not create and save it
import pickle
import os



# print(f"Time taken to convert train dataset to conversation format: {end_time - start_time} seconds")
# print("Converting val dataset to conversation format")
# val_converted_dataset = [convert_to_conversation(sample) for sample in val_dataset]
# print(f"Train dataset length: {len(train_converted_dataset)}")
# print(f"Val dataset length: {len(val_converted_dataset)}")
# st()

from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported
FastVisionModel.for_training(model) # Enable for training!

# Move model to GPU and wrap with DDP if distributed
model = model.to(f'cuda:{local_rank}')
if world_size > 1:
    # For MoE models, we need find_unused_parameters=True because not all experts are used in every forward pass
    # However, this can cause "ready twice" errors for shared experts that are always used
    # We use broadcast_buffers=False to reduce synchronization overhead and potential conflicts
    # gradient_as_bucket_view=True helps optimize gradient synchronization
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank, 
        # find_unused_parameters=True,
        # gradient_as_bucket_view=True,
        # broadcast_buffers=False  # Helps avoid synchronization issues with MoE and shared experts
    )
    # NOTE: We cannot use _set_static_graph() with MoE models because:
    # 1. MoE routing selects different experts each iteration, making the graph dynamic
    # 2. Static graph requires the same parameters to receive gradients every iteration
    # 3. This is fundamentally incompatible with MoE's dynamic expert selection
    # Instead, we rely on find_unused_parameters=True to handle unused parameters correctly

data_collator = DeepSeekOCRDataCollator(
    tokenizer=tokenizer,
    model = model.module if world_size > 1 else model,  # Use .module for DDP wrapped model
    # image_size=640,
    # base_size=1024,
    image_size=512,
    base_size=512,    
    crop_mode=True,
    train_on_responses_only=True,
)
# st()

# Initialize wandb only on main process
if is_main_process:
    wandb.init(project="dpsk-ocr-finetuning")
    experiment_name = wandb.run.name
else:
    experiment_name = "distributed_run"
# experiment_name = "revived-totem-54"
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator, # Must use!
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 500,
        # max_steps = 60,
        
        
        num_train_epochs = 20, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        eval_steps = 250,
        eval_strategy = "steps",
        save_steps = 250,
        save_strategy = "best",
        load_best_model_at_end = True,
        metric_for_best_model="eval_loss",
        # save_strategy = "steps",
        save_total_limit=1,
        # optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        output_dir = f"/grogu/user/mprabhud/dpsk_ckpts/{experiment_name}",
        # report_to = "none",     # For Weights and Biases
        dataloader_num_workers=2,
        report_to="wandb" if is_main_process else "none",  # Only report to wandb on main process
        gradient_checkpointing_kwargs={"use_reentrant": False}, # Example: explicitly set use_reentrant
        # use_reentrant=False,
        # Distributed training settings
        # For MoE models, we need find_unused_parameters=True because not all experts are used in every forward pass
        # This must match the DDP wrapper setting
        # ddp_find_unused_parameters=True,
        dataloader_pin_memory=False,  # Can help with distributed training
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        label_names = ["labels"],
    ),
)

# Load checkpoint if it exists
checkpoint_path = f"/grogu/user/mprabhud/dpsk_ckpts/{experiment_name}"
resume_from_checkpoint = None
if os.path.exists(checkpoint_path):
    # Find the latest checkpoint
    checkpoints = [d for d in os.listdir(checkpoint_path) if d.startswith("checkpoint-")]
    if checkpoints:
        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        latest_checkpoint = os.path.join(checkpoint_path, checkpoints[-1])
        if is_main_process:
            print(f"Found checkpoint: {latest_checkpoint}")
        resume_from_checkpoint = latest_checkpoint

# import ipdb; ipdb.set_trace()
if experiment_name != "distributed_run" and resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    trainer.train()

# Clean up distributed training
if world_size > 1:
    dist.destroy_process_group()
