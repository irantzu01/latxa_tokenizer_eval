import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from evaluation_helper_functions import (
    dynamic_tokenize_texts,
    dynamic_tokens_to_latxa_ids,
    build_batch_tensors,
    score_choices
)

from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import random
import torch
from tqdm import tqdm

# ==================== CONFIGURATION ====================
# Change these paths to evaluate different models
MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final")
MODEL_NAME = "250k"  # Used in output filenames

# Alternative models to test:
# MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_100k_improved/checkpoint-epoch2")
# MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_500k_improved/final")
# MODEL_PATH = "HiTZ/latxa-7b-v1.2"  # Original baseline

print(f"\n{'='*60}")
print(f"Evaluating model: {MODEL_NAME}")
print(f"Model path: {MODEL_PATH}")
print(f"{'='*60}\n")

# ==================== LOAD DATASET ====================
ds = load_dataset("HiTZ/EusProficiency", split="test")

# ==================== LOAD MODEL AND TOKENIZER ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading realigned model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()
print(f"✓ Model loaded: {MODEL_NAME}")

# ==================== HELPER FUNCTIONS ====================
def build_doc_text(item):
    return (
        f"Galdera: {item['question']}\n"
        f"A: {item['candidates'][0]}\n"
        f"B: {item['candidates'][1]}\n"
        f"C: {item['candidates'][2]}\n"
        f"D: {item['candidates'][3]}\n"
        f"Erantzuna:"
    )

def build_fewshot_example(item):
    answer_letter = ["A", "B", "C", "D"][item["answer"]]
    return build_doc_text(item) + " " + answer_letter

def build_fewshot_context(ds, current_idx, k=5, seed=42):
    rng = random.Random(seed + current_idx)
    # pool excludes current example
    pool = [ds[i] for i in range(len(ds)) if i != current_idx]
    fewshot_examples = rng.sample(pool, k)
    texts = [build_fewshot_example(ex) for ex in fewshot_examples]
    return "\n\n".join(texts)

# ==================== TOKENIZATION ====================
CHOICES = [" A", " B", " C", " D"]

pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id

# Create cache directory if it doesn't exist
os.makedirs("cache", exist_ok=True)

cache_file = f"cache/eusproficiency_{MODEL_NAME}_fewshot_tokenized.jsonl"
queries_file = f"cache/eusproficiency_{MODEL_NAME}_queries.jsonl"

print(f"\nTokenizing dataset with new tokenizer...")
print(f"Cache file: {cache_file}")

with open(cache_file, "w") as f, \
     open(queries_file, "w") as f2:
    for idx, item in enumerate(tqdm(ds, desc=f"Tokenizing ({MODEL_NAME} few-shot)")):
        fewshot_context = build_fewshot_context(ds, idx, k=5)
        query_text = build_doc_text(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        prompt_tokens = tokenizer.tokenize(prompt_text)
        query_tokens = tokenizer.tokenize(query_text) 
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        
        choice_tokens = {
            c: tokenizer.tokenize(" " + c)
            for c in ["A", "B", "C", "D"]
        }
        
        f.write(json.dumps({
            "id": idx,
            "prompt_tokens": prompt_tokens,
            "choice_tokens": choice_tokens,
            "gold": item["answer"]
        }) + "\n")
        
        f2.write(json.dumps({
            "id": idx,
            "query_tokens": query_tokens,
            "query_ids": query_ids,
            "gold": item["answer"]
        }) + "\n")

print(f"✓ Tokenization cached")

# ==================== EVALUATION ====================
correct = 0
total = 0

results_path = f"cache/eusproficiency_{MODEL_NAME}_eval_results.jsonl"

print(f"\nEvaluating model...")
print(f"Results will be saved to: {results_path}")

with open(cache_file) as fin, \
     open(results_path, "w") as fout:

    for line in tqdm(fin, desc=f"Evaluating ({MODEL_NAME} few-shot)"):

        item = json.loads(line)

        # ----- Reconstruct prompt -----
        prompt_text = tokenizer.convert_tokens_to_string(
            item["prompt_tokens"]
        )

        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False
        )

        # ----- Build full sequences -----
        full_ids = []
        for c in ["A", "B", "C", "D"]:
            choice_text = tokenizer.convert_tokens_to_string(
                item["choice_tokens"][c]
            )

            choice_ids = tokenizer.encode(
                choice_text,
                add_special_tokens=False
            )

            full_ids.append(prompt_ids + choice_ids)

        # ----- Batch + score -----
        input_ids, attention_mask = build_batch_tensors(
            full_ids, pad_id, device
        )

        scores = score_choices(model, input_ids, attention_mask)
        pred = torch.argmax(scores).item()
        is_correct = (pred == item["gold"])

        # ----- Accumulate -----
        if is_correct:
            correct += 1
        total += 1

        # ----- Save instance result -----
        fout.write(json.dumps({
            "id": item["id"],
            "gold": item["gold"],
            "prediction": pred,
            "correct": is_correct,
            "scores": scores.tolist()
        }) + "\n")

accuracy = correct / total

print(f"\n{'='*60}")
print(f"Results for: {MODEL_NAME}")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Results saved to: {results_path}")
print(f"{'='*60}\n")
