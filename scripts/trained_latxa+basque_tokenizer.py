#!/usr/bin/env python3
"""
Lexical realignment for Latxa 7B using a new Basque tokenizer.

- Freezes all middle layers
- Only trains input embeddings and LM head
- Trains on HPLT 10% + Wikipedia + Egunkaria from Hugging Face datasets
"""

# ================== 0️⃣ Imports ==================
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from tqdm import tqdm
import math
import os

# ================== 1️⃣ Settings ==================
model_name = "HiTZ/latxa-7b-v1.2"
tokenizer_dir = "basque_tokenizer_hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4                  # Adjust for GPU memory
gradient_accumulation_steps = 8
learning_rate = 1e-4
epochs = 10
max_length = 1024               # Start smaller for stability
save_dir = "latxa7b_basque_aligned"

os.makedirs(save_dir, exist_ok=True)

# ================== 2️⃣ Load tokenizer and model ==================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

# Set pad token if missing (LLaMA-based tokenizers often lack it)
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

# ================== 4️⃣ Streaming Dataset ==================
class StreamingBasqueDataset(Dataset):
    """Streams lines from file, tokenizes on the fly"""
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Count number of lines
        with open(file_path, "r", encoding="utf-8") as f:
            self.n_lines = sum(1 for _ in f)

    def __len__(self):
        return self.n_lines

    def __getitem__(self, idx):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    line = line.strip()
                    break
            else:
                line = ""
        enc = self.tokenizer(
            line,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}

# Collate function for padding
def collate_fn(batch):
    return tokenizer.pad(batch, padding=True, return_tensors="pt")

# Load dataset
full_dataset = StreamingBasqueDataset("data/basque_corpus.txt", tokenizer, max_length=max_length)

# Split for validation
val_fraction = 0.01
n_val = max(1, int(len(full_dataset) * val_fraction))
n_train = len(full_dataset) - n_val
train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ================== 5️⃣ Optimizer & Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
num_training_steps = epochs * len(train_loader) // gradient_accumulation_steps
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ================== 6️⃣ Perplexity evaluation ==================
@torch.no_grad()
def evaluate_ppl(model, dataloader):
    if len(dataloader) == 0:
        return float("nan")
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        losses.append(outputs.loss.item())
    model.train()
    return math.exp(sum(losses) / len(losses))

# ================== 7️⃣ Training loop ==================
model.train()
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

        loop.set_postfix(loss=loss.item() * gradient_accumulation_steps)

    # Evaluate perplexity
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