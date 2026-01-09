#!/usr/bin/env python3
"""
Lexical realignment for Latxa 7B using a new Basque tokenizer.

- Freezes all middle layers
- Only trains input embeddings and LM head
- Trains on HPLT 10% + Wikipedia + Egunkaria from Hugging Face datasets
"""

# ================== 0️⃣ Imports ==================
import os
import torch
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
model_name = "HiTZ/latxa-7b-v1.2"
tokenizer_dir = "basque_tokenizer_hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4                      # Adjust per GPU memory
gradient_accumulation_steps = 8     # Effective batch size = batch_size * grad_accum
learning_rate = 1e-4
epochs = 3
max_length = 1024
save_dir = "latxa7b_basque_aligned"
corpus_file = "data/basque_corpus.txt"
val_fraction = 0.01                  # Fraction of corpus for validation

os.makedirs(save_dir, exist_ok=True)

# ================== 2️⃣ Load tokenizer ==================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

# ================== 3️⃣ Load Latxa model ==================
print("Loading Latxa 7B model...")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)

# ================== 4️⃣ Embed alignment ==================
print("Aligning embeddings with Basque tokenizer...")
latxa_vocab = AutoTokenizer.from_pretrained(model_name).get_vocab()
basque_vocab = tokenizer.get_vocab()

old_token_ids = [basque_vocab[tok] for tok in basque_vocab if tok in latxa_vocab]
new_token_ids = [basque_vocab[tok] for tok in basque_vocab if tok not in latxa_vocab]

model.resize_token_embeddings(len(tokenizer))

# Initialize new embeddings randomly
embedding_weights = model.get_input_embeddings().weight.data
for idx in new_token_ids:
    embedding_weights[idx] = torch.randn(model.config.hidden_size) * 0.02

# Freeze middle layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LM head
model.get_output_embeddings().weight.requires_grad = True

# Unfreeze embeddings
model.get_input_embeddings().weight.requires_grad = True

# Freeze old embeddings using a hook
old_idx_tensor = torch.tensor(old_token_ids, dtype=torch.long)
def zero_grad_old_tokens(grad):
    grad.index_fill_(0, old_idx_tensor.to(grad.device), 0)
    return grad
model.get_input_embeddings().weight.register_hook(zero_grad_old_tokens)

print(f"New tokens: {len(new_token_ids)}, old tokens frozen: {len(old_token_ids)}")
print("Middle layers frozen. Only new token embeddings + LM head will be trained.")

model = model.to(device)

# ================== 5️⃣ Streaming Dataset ==================
class BasqueStreamingDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_length=1024, val_fraction=0.01, split="train"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.val_fraction = val_fraction
        self.split = split

        # Count total lines
        with open(file_path, "r", encoding="utf-8") as f:
            self.total_lines = sum(1 for _ in f)
        self.train_cutoff = int(self.total_lines * (1 - val_fraction))

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

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

train_dataset = BasqueStreamingDataset(corpus_file, tokenizer, max_length=max_length, split="train")
val_dataset = BasqueStreamingDataset(corpus_file, tokenizer, max_length=max_length, split="val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# ================== 6️⃣ Optimizer & Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

num_training_steps = epochs * (len(train_loader) // gradient_accumulation_steps)
num_warmup_steps = int(0.05 * num_training_steps)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ================== 7️⃣ Validation perplexity ==================
@torch.no_grad()
def evaluate_ppl(model, dataloader, max_batches=50):
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        losses.append(outputs.loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses)) if losses else float("nan")

# ================== 8️⃣ Training loop ==================
model.train()
global_step = 0

for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    for step, batch in enumerate(loop):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        loop.set_postfix(loss=loss.item() * gradient_accumulation_steps)

    ppl = evaluate_ppl(model, val_loader)
    print(f"Epoch {epoch+1} completed. Validation perplexity: {ppl:.2f}")

# ================== 9️⃣ Save model ==================
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Lexically realigned Latxa 7B saved to '{save_dir}'")

# ================== 🔟 Test generation ==================
model.eval()
test_sentence = "Euskal Herria da gure herria."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated:", decoded)



