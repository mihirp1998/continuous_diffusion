from itertools import chain
import warnings
import math
import torch
from transformers import TrainerCallback
import wandb
import ipdb
st = ipdb.set_trace
#from hf
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import TrainingArguments, Trainer

import pickle
import os

data_dir = os.environ["DATA_DIR"]

# Load TinyStories dataset
print("Loading TinyStories dataset from cache...")
with open(f"{data_dir}/stories_cache_500000_eval_False.pkl", "rb") as f:
    train_stories = pickle.load(f)

# Load validation set from original dataset
print("Loading validation set...")
val_dataset = load_dataset("roneneldan/TinyStories", split="validation[:1000]")
val_stories = val_dataset["text"]
# Create dataset dictionary using datasets.Dataset
from datasets import Dataset

train_dataset = Dataset.from_dict({"text": train_stories})
val_dataset = Dataset.from_dict({"text": val_stories})

dataset = {
    "train": train_dataset,
    "validation": val_dataset
}

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=1024)

print("Tokenizing dataset...")
tokenized_datasets = {
    "train": dataset["train"].map(tokenize_function, batched=True, remove_columns=["text"]),
    "validation": dataset["validation"].map(tokenize_function, batched=True, remove_columns=["text"])
}
# /
# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked language modeling
)

# Initialize GPT-2 model from scratch
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    resid_pdrop=0.1,
    embd_pdrop=0.1,
    attn_pdrop=0.1,
)


print("Initializing GPT-2 model from scratch...")
model = GPT2LMHeadModel(config)
# Text generation callback
class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts=None, generation_steps=1000):
        self.tokenizer = tokenizer
        self.prompts = prompts or ["Once upon a time", "The little girl", "In a magical forest", "Once"]
        self.generation_steps = generation_steps
        self.vis_dict = []
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # st()
        if state.global_step % self.generation_steps == 0 and state.global_step > 0:
            model.eval()
            print(f"\n--- Text Generation at Step {state.global_step} ---")
            
            # Create wandb table for this step
            
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated_text}")
                print("-" * 50)
                
                # Add to table data
                self.vis_dict.append([prompt, generated_text, state.global_step])
            
            # Create wandb table and log it
            table = wandb.Table(columns=["Prompt", "Generated Text", "Step"], data=self.vis_dict)
            wandb.log({"generated_text_table": table}, step=state.global_step)
            
            
            model.train()


class GenerativePerplexityCallback(TrainerCallback):
    def __init__(self, tokenizer, prompts=None, generation_steps=1000):
        self.tokenizer = tokenizer
        self.prompts = prompts or ["Once upon a time", "The little girl", "In a magical forest", "Once"]
        self.generation_steps = generation_steps
        
        # Load pretrained GPT-2 model for perplexity evaluation
        self.eval_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.eval_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.generation_steps == 0 and state.global_step > 0:
            model.eval()
            self.eval_model.eval()
            # st()
            
            # Move eval model to same device as training model
            self.eval_model.to(model.device)
            
            total_perplexity = 0
            num_samples = 0
            
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    # Generate text using our training model
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Get the generated text
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Evaluate perplexity using pretrained GPT-2
                    eval_inputs = self.eval_tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
                    
                    # Get model outputs for perplexity calculation
                    eval_outputs = self.eval_model(**eval_inputs, labels=eval_inputs.input_ids)
                    loss = eval_outputs.loss
                    
                    # Calculate perplexity (exp of loss)
                    perplexity = torch.exp(loss).item()
                    total_perplexity += perplexity
                    num_samples += 1
            
            # Calculate average perplexity
            avg_perplexity = total_perplexity / num_samples
            
            print(f"Average Generative Perplexity (GPT-2 eval) at Step {state.global_step}: {avg_perplexity:.4f}")
            
            # Log to wandb
            wandb.log({"generative_perplexity_gpt2": avg_perplexity}, step=state.global_step)
            
            model.train()


# Training arguments
training_args = TrainingArguments(
    output_dir="./tinystories-gpt2",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    prediction_loss_only=True,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_num_workers=4,
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    report_to="wandb",
)

# Initialize text generation callback
text_gen_callback = TextGenerationCallback(tokenizer)
perplexity_callback = GenerativePerplexityCallback(tokenizer)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    callbacks=[text_gen_callback, perplexity_callback],
)

# Initialize wandb
wandb.init(project="tinystories-gpt2")

# Start training
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model()
print("Training completed and model saved!")
