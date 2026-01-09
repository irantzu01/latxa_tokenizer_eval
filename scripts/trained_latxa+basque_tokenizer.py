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
from torch.utils.data import DataLoader, Dataset
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

batch_size = 4                     # Adjust for GPU memory
gradient_accumulation_steps = 8    # Effective batch size = batch_size * grad_accum
learning_rate = 1e-4
epochs = 3                         # Increase later if needed
max_length = 1024
save_dir = "latxa7b_basque_aligned"

val_fraction = 0.01                 # fraction of corpus for validation
approx_num_examples = 200_000_000   # estimate of sentences in corpus

corpus_file = "data/basque_corpus.txt"

os.makedirs(save_dir, exist_ok=True)

# ================== 2️⃣ Load tokenizer ==================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")

# ================== 3️⃣ Load Latxa model ==================
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", low_cpu_mem_usage=True
)

# Resize embeddings to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# ================== 4️⃣ Freeze middle layers ==================
for name, param in model.named_parameters():
    param.requires_grad = False

# Only train input embeddings + LM head
model.get_input_embeddings().weight.requires_grad = True
model.get_output_embeddings().weight.requires_grad = True
print("Middle layers frozen. Only input embeddings and LM head will be trained.")

# ================== 5️⃣ Dataset ==================
class BasqueCorpusDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024, val_fraction=0.01):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load lines from file
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(line)

        # Split into train/validation
        split_idx = int(len(self.examples) * (1 - val_fraction))
        self.train_examples = self.examples[:split_idx]
        self.val_examples = self.examples[split_idx:]

    def get_train_dataset(self):
        return self._tokenized_dataset(self.train_examples)

    def get_val_dataset(self):
        return self._tokenized_dataset(self.val_examples)

    def _tokenized_dataset(self, lines):
        dataset = []
        for line in lines:
            enc = self.tokenizer(
                line,
                truncation=True,
                max_length=self.max_length,
            )
            dataset.append({
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long)
            })
        return dataset

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attention_mask"] for x in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}

# Create datasets and loaders
corpus_dataset = BasqueCorpusDataset(corpus_file, tokenizer, max_length=max_length, val_fraction=val_fraction)
train_loader = DataLoader(corpus_dataset.get_train_dataset(), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(corpus_dataset.get_val_dataset(), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ================== 6️⃣ Optimizer + Scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

effective_batch_size = batch_size * gradient_accumulation_steps
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
    if len(losses) == 0:
        return float("nan")
    return math.exp(sum(losses) / len(losses))

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

    # Evaluate validation PPL after each epoch
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

