import sys
import os
import json



project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)


from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from zett.utils import get_surface_form_matrix
from collections import Counter
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch



def format_prompt(question, candidates):
    return (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )


#Load the EusProficiency dataset and prepare evaluation items
ds = load_dataset("HiTZ/EusProficiency", split="test")

CHOICES = [" A", " B", " C", " D"]
evaluation_items = []
for item in ds:
    prompt = format_prompt(item["question"], item["candidates"])
    
    choice_texts = [prompt + choice for choice in CHOICES]

    evaluation_items.append({
        "prompt": prompt,
        "choice_texts": choice_texts,   # length 4
        "answer": item["answer"]        # int: 0–3
    })
print(evaluation_items[0])
print("Evaluation items prepared:", len(evaluation_items))


# Load hypernetwork
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental",
    trust_remote_code=True
)
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)
dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)
print("Hypernetwork + tokenizer + Dynamic BPE ready.")



def dynamic_tokenize_texts(texts, dynamic_bpe, batch_size=128, max_merges=10):
    """
    texts: list[str]
    returns: list[list[str]]  (dynamic tokens per text)
    """
    all_tokens = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Dynamic BPE"):
        batch_texts = texts[i:i+batch_size]
        batch_examples = [{"text": t} for t in batch_texts]

        dyn_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
            batch_examples=batch_examples,
            max_nr_merges=max_merges,
            mlm=True
        )

        all_tokens.extend(dyn_tokens)

    return all_tokens

for item in evaluation_items:
    dynamic_choice_tokens = dynamic_tokenize_texts(
        item["choice_texts"],
        dynamic_bpe,
        batch_size=4
    )
    # Ensure structure: list[list[str]]
    assert isinstance(dynamic_choice_tokens, list)
    assert isinstance(dynamic_choice_tokens[0], list)
    item["dynamic_tokens"] = dynamic_choice_tokens

print("Dynamic tokenization completed.")



from dynamic_augmenter import DynamicAugmenter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load Latxa tokenizer and model
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
model = model.to(device)
print("Latxa model and tokenizer loaded.")

# Initialize DynamicAugmenter
augmenter = DynamicAugmenter(
    model=model,
    latxa_tokenizer=latxa_tokenizer,
    hypernet=hypernet,
    hypernet_tokenizer=hypernet_tokenizer,
    cache_limit=50_000,   # safe default
    device=device
)
print("DynamicAugmenter ready.")


# Convert dynamic tokens → token IDs using DynamicAugmenter
all_choice_token_ids = []

for item in tqdm(evaluation_items, desc="Mapping dynamic tokens to IDs"):
    choice_token_ids = augmenter.tokens_to_ids(item["dynamic_tokens"])
    all_choice_token_ids.append(choice_token_ids)

print("Dynamic token → ID conversion completed.")


# Build batch tensors
def build_batch_tensors(batch_ids, pad_id, device):
    """
    batch_ids: list[list[int]]  (len = 4 choices)
    """
    max_len = max(len(seq) for seq in batch_ids)

    input_ids = torch.full(
        (len(batch_ids), max_len),
        pad_id,
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.zeros_like(input_ids)

    for i, seq in enumerate(batch_ids):
        seq = torch.tensor(seq, dtype=torch.long, device=device)
        input_ids[i, :len(seq)] = seq
        attention_mask[i, :len(seq)] = 1

    return input_ids, attention_mask


# Multiple-choice scoring (log-likelihood of last token)
@torch.no_grad()
def score_choices(model, input_ids, attention_mask):
    """
    input_ids: (4, seq_len)
    Returns: tensor of shape (4,) with log-likelihood scores
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    logits = outputs.logits  # (4, seq_len, vocab_size)

    last_token_positions = attention_mask.sum(dim=1) - 1
    scores = []

    for i in range(input_ids.size(0)):
        pos = last_token_positions[i]
        token_id = input_ids[i, pos]
        log_probs = torch.log_softmax(logits[i, pos], dim=-1)
        scores.append(log_probs[token_id])

    return torch.stack(scores)


# Evaluation loop
pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id

correct = 0
total = 0

model.eval()

for item, choice_ids in tqdm(
    zip(evaluation_items, all_choice_token_ids),
    total=len(evaluation_items),
    desc="Evaluating"
):
    input_ids, attention_mask = build_batch_tensors(
        choice_ids,
        pad_id,
        device
    )

    scores = score_choices(model, input_ids, attention_mask)
    predicted = torch.argmax(scores).item()

    if predicted == item["answer"]:
        correct += 1
    total += 1


accuracy = correct / total
print(f"\nFinal accuracy (Dynamic BPE + Hypernet): {accuracy:.4f}")
