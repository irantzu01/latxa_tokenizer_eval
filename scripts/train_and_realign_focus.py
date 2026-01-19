#!/usr/bin/env python3

import os
import sys
import torch
import shutil
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler)
from tqdm import tqdm
import math
from focus_initialization import initialize_embeddings_with_focus


# ================ Settings ==================
model_name = "HiTZ/latxa-7b-v1.2"
tokenizer_dir = "basque_tokenizer_hf"

# Check GPU availability
if torch.cuda.is_available():
    device = "cuda"
    print(f"\n{'='*60}")
    print(f"GPU detected!")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
else:
    device = "cpu"
    print("\n⚠️  WARNING: No GPU detected! Training will be VERY slow on CPU.")
    print("Make sure to request GPU in your SLURM job with: #SBATCH --gres=gpu:A100:1\n")

batch_size = 8                      
gradient_accumulation_steps = 4     
learning_rate = 5e-4                
epochs = 3
max_length = 1024
save_dir = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved")
corpus_file = "data/basque_corpus_sampled_250k.txt"
val_fraction = 0.01                  
max_steps_per_epoch = None           

os.makedirs(save_dir, exist_ok=True)

# ================ Load tokenizer ==================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================ Load Latxa model ==================
print("Loading Latxa 7B model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16
)

# ================ Embed alignment with FOCUS ==================
print("Aligning embeddings with Basque tokenizer using FOCUS...")
latxa_tokenizer = AutoTokenizer.from_pretrained(model_name)

# FOCUS will handle resizing internally - don't resize here!
# Initialize embeddings with FOCUS method
old_token_ids, new_token_ids = initialize_embeddings_with_focus(
    model=model,
    old_tokenizer=latxa_tokenizer,
    new_tokenizer=tokenizer,
    device=device
)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LM head
for param in model.get_output_embeddings().parameters():
    param.requires_grad = True

# Unfreeze all embeddings
model.get_input_embeddings().weight.requires_grad = True

print("Middle layers frozen. All token embeddings + LM head will be trained.")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ================== Streaming Dataset ==================
class BasqueStreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=1024, val_fraction=0.01, split="train"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_fraction = val_fraction
        self.split = split

        print(f"Counting lines in {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            self.total_lines = sum(1 for _ in f)
        print(f"Total lines: {self.total_lines:,}")
        
        self.train_cutoff = int(self.total_lines * (1 - val_fraction))
        print(f"Train lines: {self.train_cutoff:,}, Val lines: {self.total_lines - self.train_cutoff:,}")

    def __iter__(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                if self.split == "train" and idx >= self.train_cutoff:
                    continue
                if self.split == "val" and idx < self.train_cutoff:
                    continue
                
                enc = self.tokenizer(
                    line,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                yield {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0)
                }
    
    def get_num_samples(self):
        if self.split == "train":
            return self.train_cutoff
        else:
            return self.total_lines - self.train_cutoff

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Create datasets
train_dataset = BasqueStreamingDataset(corpus_file, tokenizer, max_length=max_length, split="train")
val_dataset = BasqueStreamingDataset(corpus_file, tokenizer, max_length=max_length, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Estimate steps per epoch
estimated_train_samples = train_dataset.get_num_samples()
estimated_steps_per_epoch = estimated_train_samples // batch_size
if max_steps_per_epoch:
    estimated_steps_per_epoch = min(estimated_steps_per_epoch, max_steps_per_epoch)

print(f"\nEstimated steps per epoch: {estimated_steps_per_epoch:,}")
print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")

# ================== Optimizer & Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

num_training_steps = epochs * (estimated_steps_per_epoch // gradient_accumulation_steps)
num_warmup_steps = int(0.1 * num_training_steps)
print(f"Total training steps: {num_training_steps:,}")
print(f"Warmup steps: {num_warmup_steps:,}")

scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ================== Validation perplexity ==================
@torch.no_grad()
def evaluate_ppl(model, dataloader, max_batches=100):
    model.eval()
    losses = []
    pbar = tqdm(dataloader, desc="Validating", total=max_batches)
    for i, batch in enumerate(pbar):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        losses.append(outputs.loss.item())
        pbar.set_postfix({"loss": outputs.loss.item()})
    
    model.train()
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    perplexity = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")
    return perplexity, avg_loss

# ================== Training loop ==================
print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

model.train()
global_step = 0
best_ppl = float('inf')
best_epoch = 0

for epoch in range(epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*50}")
    
    epoch_steps = 0
    epoch_loss = 0
    optimizer.zero_grad()
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=estimated_steps_per_epoch)
    
    for step, batch in enumerate(loop):
        if max_steps_per_epoch and step >= max_steps_per_epoch:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        epoch_loss += outputs.loss.item()
        epoch_steps += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        loop.set_postfix({
            'loss': f'{outputs.loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}',
            'step': global_step
        })
    
    # Final gradient accumulation step if needed
    if epoch_steps % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1
    
    avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else float('nan')
    print(f"\nEpoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
    
    # Validation
    print("\nRunning validation...")
    val_ppl, val_loss = evaluate_ppl(model, val_loader, max_batches=100)
    print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
    
    # Save best model to 'final' directory
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_epoch = epoch + 1
        
        final_dir = os.path.join(save_dir, "final")
        
        # Remove old best checkpoint if it exists
        if os.path.exists(final_dir):
            print(f"Removing previous best checkpoint...")
            shutil.rmtree(final_dir)
        
        # Save new best
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        
        # Save metadata
        with open(os.path.join(final_dir, "best_checkpoint_info.txt"), "w") as f:
            f.write(f"Best checkpoint from Epoch {best_epoch}\n")
            f.write(f"Validation Loss: {val_loss:.4f}\n")
            f.write(f"Validation Perplexity: {val_ppl:.2f}\n")
        
        print(f"✓ New best model saved to {final_dir} (Epoch {best_epoch}, PPL: {val_ppl:.2f})")
    else:
        print(f"  No improvement (best PPL: {best_ppl:.2f} from Epoch {best_epoch})")

# ================== Training complete ==================
print(f"\n{'='*50}")
print(f"Training complete!")
print(f"Best model saved to '{os.path.join(save_dir, 'final')}'")
print(f"Best validation perplexity: {best_ppl:.2f} (Epoch {best_epoch})")
print(f"{'='*50}\n")

# ================== Test generation ==================
print("Testing generation...")
model.eval()
test_sentences = [
    "Euskal Herria da gure herria.",
    "Gaur egun, teknologia",
    "Bilbo hiriak"
]

for test_sentence in test_sentences:
    inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInput: {test_sentence}")
    print(f"Generated: {decoded}")