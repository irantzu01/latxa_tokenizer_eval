#!/usr/bin/env python3
"""
Lexical realignment for Latxa 7B using a new Basque tokenizer.

- Freezes all middle layers
- Only trains input embeddings and LM head
- Trains on HPLT 10% + Wikipedia + Egunkaria from Hugging Face datasets
"""

# ================== 0️⃣ Imports ==================
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import math
import os

# ================== 1️⃣ Settings ==================
model_name = "HiTZ/latxa-7b-v1.2"
tokenizer_dir = "basque_tokenizer_hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 4                     # Adjust for GPU memory
gradient_accumulation_steps = 8    # Effective batch size = batch_size * grad_accum
learning_rate = 1e-4
epochs = 3                         # Adjust as needed
max_length = 1024                   # Can increase later
save_dir = "latxa7b_basque_aligned"

approx_num_examples = 200_000_000   # Estimate of corpus size for scheduler
val_fraction = 0.01                 # 1% for validation

os.makedirs(save_dir, exist_ok=True)

# ================== 2️⃣ Load tokenizer and model ==================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# ================== 3️⃣ Freeze middle layers ==================
for name, param in model.named_parameters():
    param.requires_grad = False

model.get_input_embeddings().weight.requires_grad = True
model.get_output_embeddings().weight.requires_grad = True
print("Middle layers frozen. Only input embeddings and LM head will be trained.")

# ================== 4️⃣ Load streaming dataset ==================
dataset = load_dataset("text", data_files="data/basque_corpus.txt", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=10_000)  # small shuffle buffer for randomness

# Split 1% for validation
val_dataset = dataset.take(int(val_fraction * approx_num_examples))
train_dataset = dataset.skip(int(val_fraction * approx_num_examples))

# Tokenization function
def tokenize_fn(batch):
    enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
    return enc

train_dataset = train_dataset.map(tokenize_fn)
val_dataset = val_dataset.map(tokenize_fn)

# Collate function for DataLoader
def collate_fn(batch):
    return tokenizer.pad(batch, padding=True, return_tensors="pt")

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# ================== 5️⃣ Optimizer & Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

effective_batch_size = batch_size * gradient_accumulation_steps
num_training_steps = epochs * (approx_num_examples // effective_batch_size)
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ================== 6️⃣ Validation perplexity ==================
@torch.no_grad()
def evaluate_ppl(model, dataloader, max_batches=50):
    """Evaluate PPL on up to max_batches batches from validation set."""
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        losses.append(outputs.loss.item())
    model.train()
    if len(losses) == 0:
        return float("nan")
    return math.exp(sum(losses) / len(losses))

# ================== 7️⃣ Training loop ==================
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

# ================== 8️⃣ Save aligned model ==================
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Lexically realigned Latxa 7B saved to '{save_dir}'")

# ================== 9️⃣ Test generation ==================
model.eval()
test_sentence = "Euskal Herria da gure herria."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated:", decoded)

