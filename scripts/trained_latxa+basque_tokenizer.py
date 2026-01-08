#!/usr/bin/env python3
"""
Lexical realignment for Latxa 7B using a new Basque tokenizer.

- Freezes all middle layers
- Only trains input embeddings and LM head
- Trains on HPLT 10% + Wikipedia + Egunkaria from Hugging Face datasets
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import os

# ================== 1️⃣ Settings ==================
model_name = "HiTZ/latxa-7b-v1.2"
tokenizer_dir = "basque_tokenizer_hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2                # Adjust for GPU memory
learning_rate = 5e-5
epochs = 1                    # You can increase depending on dataset size
max_length = 2048
save_dir = "latxa7b_basque_aligned"

os.makedirs(save_dir, exist_ok=True)

# ================== 2️⃣ Load tokenizer and model ==================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", low_cpu_mem_usage=True)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# ================== 3️⃣ Freeze middle layers ==================
for name, param in model.named_parameters():
    param.requires_grad = False  # Freeze everything first

# Unfreeze input embeddings and output head
model.get_input_embeddings().weight.requires_grad = True
model.get_output_embeddings().weight.requires_grad = True

print("Middle layers frozen. Only input embeddings and LM head will be trained.")


# ================== 4️⃣ Load and prepare local corpus ==================
class BasqueCorpusDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Read the file line by line
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(line)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Tokenize each line on the fly
        enc = self.tokenizer(
            self.examples[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
    

def collate_fn(batch):
    # batch is a list of dicts: {"input_ids": ..., "attention_mask": ...}
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    # Pad to the max length in the batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or "<s>" if you prefer
    print(f"Pad token set to: {tokenizer.pad_token} ({tokenizer.pad_token_id})")


# Create the dataset and DataLoader
train_dataset = BasqueCorpusDataset("data/basque_corpus.txt", tokenizer, max_length=2048)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# ================== 5️⃣ Optimizer and scheduler ==================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
num_training_steps = epochs * len(train_loader)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# ================== 6️⃣ Training loop ==================
model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

# ================== 7️⃣ Save aligned model ==================
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Lexically realigned Latxa 7B saved to '{save_dir}'")

# ================== 8️⃣ Test generation ==================
model.eval()
test_sentence = "Euskal Herria da gure herria."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated:", decoded)
