#!/usr/bin/env python3
"""
Lexical realignment for Latxa 7B using a new Basque tokenizer.

- Freezes all middle layers
- Only trains input embeddings and LM head
- Trains on HPLT 10% + Wikipedia + Egunkaria from Hugging Face datasets
"""

# ================== 0️⃣ Imports ==================
import os
import sys
import torch
import shutil
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AdamW,
    get_scheduler
)
from tqdm import tqdm
import math

# ================== 1️⃣ Settings ==================
# Allow overriding via command line arguments
# Usage: python train.py [corpus_file] [output_suffix]
# Example: python train.py data/basque_corpus_sampled_100k.txt 100k

if len(sys.argv) > 1:
    corpus_file = sys.argv[1]
    output_suffix = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(corpus_file).replace('.txt', '').replace('basque_corpus_sampled_', '')
else:
    corpus_file = "data/basque_corpus_sampled_small.txt"
    output_suffix = "default"

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
    print("Make sure to request GPU in your SLURM job with: #SBATCH --gres=gpu:1\n")

batch_size = 8                      # Increased for A100 (was 4)
gradient_accumulation_steps = 4     # Reduced since batch_size is higher (effective batch still 32)
learning_rate = 5e-4                # Increased from 1e-4 for better embedding learning
epochs = 3
max_length = 1024
save_dir = os.path.expanduser(f"~/tmp/models/latxa7b_basque_aligned_{output_suffix}")
val_fraction = 0.01                  # Fraction of corpus for validation
max_steps_per_epoch = None           # Set to a number to limit steps per epoch (e.g., 10000)
save_total_limit = 1                 # Keep only the N most recent checkpoints (None = keep all)

os.makedirs(save_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Training Configuration")
print(f"{'='*60}")
print(f"Corpus file: {corpus_file}")
print(f"Output directory: {save_dir}")
print(f"{'='*60}\n")

# ================== 2️⃣ Load tokenizer ==================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

# ================== 3️⃣ Load Latxa model ==================
print("Loading Latxa 7B model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16  # Use bfloat16 for better memory efficiency
)

# ================== 4️⃣ Embed alignment ==================
print("Aligning embeddings with Basque tokenizer...")
latxa_tokenizer = AutoTokenizer.from_pretrained(model_name)
latxa_vocab = latxa_tokenizer.get_vocab()
basque_vocab = tokenizer.get_vocab()

# Find overlapping and new tokens
old_token_ids = []
new_token_ids = []
for tok, idx in basque_vocab.items():
    if tok in latxa_vocab:
        old_token_ids.append(idx)
    else:
        new_token_ids.append(idx)

print(f"Vocab overlap: {len(old_token_ids)} tokens")
print(f"New tokens: {len(new_token_ids)} tokens")

# Resize embeddings
old_embeddings = model.get_input_embeddings()
model.resize_token_embeddings(len(tokenizer))

# Initialize new embeddings: copy from similar tokens or use mean of existing embeddings
embedding_weights = model.get_input_embeddings().weight.data
if len(new_token_ids) > 0:
    # Calculate mean and std of existing embeddings for better initialization
    existing_embeddings = embedding_weights[:len(latxa_vocab)]
    existing_mean = existing_embeddings.mean(dim=0)
    existing_std = existing_embeddings.std(dim=0).mean().item()
    
    print(f"Initializing {len(new_token_ids)} new embeddings with mean from existing tokens")
    for idx in new_token_ids:
        # Initialize with small random noise around the mean of existing embeddings
        # Make sure the random tensor is on the same device as embedding_weights
        noise = torch.randn(model.config.hidden_size, dtype=embedding_weights.dtype, device=embedding_weights.device)
        embedding_weights[idx] = existing_mean + noise * (existing_std * 0.1)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LM head
for param in model.get_output_embeddings().parameters():
    param.requires_grad = True

# Unfreeze ALL embeddings (not just new tokens)
model.get_input_embeddings().weight.requires_grad = True

# OPTION: Uncomment below to freeze old token embeddings (currently disabled for better learning)
# if len(old_token_ids) > 0:
#     old_idx_tensor = torch.tensor(old_token_ids, dtype=torch.long)
#     def zero_grad_old_tokens(grad):
#         if grad is not None:
#             grad.index_fill_(0, old_idx_tensor.to(grad.device), 0)
#         return grad
#     model.get_input_embeddings().weight.register_hook(zero_grad_old_tokens)

print("Middle layers frozen. All token embeddings + LM head will be trained.")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

# ================== 5️⃣ Streaming Dataset ==================
class BasqueStreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=1024, val_fraction=0.01, split="train"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_fraction = val_fraction
        self.split = split

        # Count total lines
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
                
                # Split logic
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
        """Estimate number of samples for this split"""
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

# ================== 6️⃣ Optimizer & Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

num_training_steps = epochs * (estimated_steps_per_epoch // gradient_accumulation_steps)
num_warmup_steps = int(0.1 * num_training_steps)  # Increased from 0.05 to 0.1
print(f"Total training steps: {num_training_steps:,}")
print(f"Warmup steps: {num_warmup_steps:,}")

scheduler = get_scheduler(
    "cosine",  # Changed from "linear" to "cosine" for better convergence
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ================== 7️⃣ Validation perplexity ==================
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

def cleanup_old_checkpoints(save_dir, save_total_limit, keep_best=True):
    """
    Keep only the most recent checkpoints and optionally the best checkpoint.
    
    Args:
        save_dir: Base directory containing checkpoints
        save_total_limit: Number of epoch checkpoints to keep (None = keep all)
        keep_best: Whether to always keep the best checkpoint
    """
    if save_total_limit is None:
        return
    
    # Find all epoch checkpoint directories
    epoch_dirs = []
    best_dir = None
    
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path):
            if item.startswith("epoch-"):
                try:
                    epoch_num = int(item.split("-")[1])
                    epoch_dirs.append((epoch_num, item_path))
                except (ValueError, IndexError):
                    continue
            elif item.startswith("checkpoint-epoch"):
                best_dir = item_path
    
    # Sort by epoch number (most recent last)
    epoch_dirs.sort(key=lambda x: x[0])
    
    # Determine which checkpoints to delete
    if len(epoch_dirs) > save_total_limit:
        dirs_to_delete = epoch_dirs[:-save_total_limit]  # Keep only the last N
        
        for epoch_num, dir_path in dirs_to_delete:
            # Don't delete the best checkpoint if keep_best is True
            if keep_best and best_dir and os.path.samefile(dir_path, best_dir):
                continue
            
            print(f"Removing old checkpoint: {dir_path}")
            shutil.rmtree(dir_path)
            
            # Free up GPU memory if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ================== 8️⃣ Training loop ==================
print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

model.train()
global_step = 0
best_ppl = float('inf')

for epoch in range(epochs):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*50}")
    
    epoch_steps = 0
    epoch_loss = 0
    optimizer.zero_grad()
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=estimated_steps_per_epoch)
    
    for step, batch in enumerate(loop):
        # Limit steps per epoch if specified
        if max_steps_per_epoch and step >= max_steps_per_epoch:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        epoch_loss += outputs.loss.item()
        epoch_steps += 1

        if (step + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
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
    
    # Save best model
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"✓ New best model saved to {checkpoint_dir}")
    
    # Save checkpoint every epoch
    epoch_dir = os.path.join(save_dir, f"epoch-{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(epoch_dir)
    tokenizer.save_pretrained(epoch_dir)
    print(f"✓ Checkpoint saved to {epoch_dir}")
    
    # Clean up old checkpoints to save space
    cleanup_old_checkpoints(save_dir, save_total_limit, keep_best=True)
    print(f"✓ Old checkpoints cleaned up (keeping last {save_total_limit})")

# ================== 9️⃣ Save final model ==================
final_dir = os.path.join(save_dir, "final")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"\n{'='*50}")
print(f"Training complete!")
print(f"Final model saved to '{final_dir}'")
print(f"Best validation perplexity: {best_ppl:.2f}")
print(f"{'='*50}\n")

# ================== 🔟 Test generation ==================
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