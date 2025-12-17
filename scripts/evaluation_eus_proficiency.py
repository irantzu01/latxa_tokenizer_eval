import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np



# Prompt formatting
def format_prompt(question, candidates):
    return (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )


# Load EusProficiency dataset
ds = load_dataset("HiTZ/EusProficiency", split="test")


# Prepare evaluation items
CHOICES = [" A", " B", " C", " D"]
evaluation_items = []

for item in ds:
    prompt = format_prompt(item["question"], item["candidates"])
    choice_texts = [prompt + choice for choice in CHOICES]

    evaluation_items.append({
        "choice_texts": choice_texts,  # list[str] length 4
        "answer": item["answer"]       # int 0–3
    })

print("Evaluation items prepared:", len(evaluation_items))

# Dynamic BPE tokenizer
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)
dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)

def dynamic_tokenize_texts(texts, dynamic_bpe, batch_size=128, max_merges=10):
    """
    texts: list[str]
    returns: list[list[str]]
    """
    all_tokens = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_examples = [{"text": t} for t in batch_texts]

        dyn_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
            batch_examples=batch_examples,
            max_nr_merges=max_merges,
            mlm=True
        )

        all_tokens.extend(dyn_tokens)

    return all_tokens


# dynamic tokens → text
def dynamic_tokens_to_text(dynamic_tokens):
    """  list[str] → str  """
    text = ""
    for tok in dynamic_tokens:
        if tok.startswith("▁") or tok.startswith("Ġ"):
            text += " " + tok[1:]
        else:
            text += tok
    return text.strip()
print("Running Dynamic BPE tokenization...")

for item in tqdm(evaluation_items, desc="Dynamic BPE"):
    dynamic_tokens = dynamic_tokenize_texts(
        item["choice_texts"],
        dynamic_bpe,
        batch_size=4
    )
    item["dynamic_tokens"] = dynamic_tokens
print("Dynamic tokenization completed.")


# Load Latxa model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
model.to(device)
model.eval()
print("Latxa model and tokenizer loaded.")


# dynamic tokens → latxa IDs
def dynamic_tokens_to_latxa_ids(dynamic_tokens_batch, tokenizer):
    """
    list[list[str]] → list[list[int]]
    """
    texts = [
        dynamic_tokens_to_text(tokens)
        for tokens in dynamic_tokens_batch
    ]

    enc = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False
    )

    return enc["input_ids"]


# Batch builder
def build_batch_tensors(batch_ids, pad_id, device):
    max_len = max(len(seq) for seq in batch_ids)
    input_ids = torch.full(
        (len(batch_ids), max_len),
        pad_id,
        dtype=torch.long,
        device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch_ids):
        seq = torch.tensor(seq, device=device)
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1

    return input_ids, attention_mask


# Scoring
@torch.no_grad()
def score_choices(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
    targets = input_ids[:, 1:]

    scores = []
    for i in range(input_ids.size(0)):
        score = 0.0
        for t in range(targets.size(1)):
            if attention_mask[i, t + 1]:
                score += log_probs[i, t, targets[i, t]]
        scores.append(score)

    return torch.stack(scores)

# Evaluation loop
pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id

correct = 0

for item in tqdm(evaluation_items, desc="Evaluating"):
    choice_ids = dynamic_tokens_to_latxa_ids(
        item["dynamic_tokens"],
        latxa_tokenizer
    )

    input_ids, attention_mask = build_batch_tensors(
        choice_ids,
        pad_id,
        device
    )

    scores = score_choices(model, input_ids, attention_mask)
    pred = torch.argmax(scores).item()

    if pred == item["answer"]:
        correct += 1

accuracy = correct / len(evaluation_items)
print(f"\nFinal accuracy (Dynamic BPE segmentation + Latxa vocab): {accuracy:.4f}")
