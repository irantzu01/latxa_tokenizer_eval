import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from evaluation_helper_functions import (
    build_batch_tensors,
    score_choices
)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import random

# ==================== CONFIGURATION ====================
# Change these paths to evaluate different models
MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final")
MODEL_NAME = "100k_improved"  # Used in output filenames

# Alternative models to test:
# MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_100k_improved/checkpoint-epoch2")
# MODEL_PATH = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_500k_improved/final")
# MODEL_PATH = "HiTZ/latxa-7b-v1.2"  # Original baseline

print(f"\n{'='*60}")
print(f"Evaluating model on EusReading: {MODEL_NAME}")
print(f"Model path: {MODEL_PATH}")
print(f"{'='*60}\n")

# ==================== LOAD DATASET ====================
ds = load_dataset("HiTZ/EusReading", split="test")

# Filter out items with insufficient candidates
valid_items = []
for item in ds:
    candidates = item["candidates"]
    candidates = [o for o in candidates if o and o.strip()]
    if len(candidates) >= 2:
        valid_items.append(item)

print(f"Total items: {len(ds)}")
print(f"Valid items (≥2 candidates): {len(valid_items)}")

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
def format_prompt_reading(context, question, candidates):
    """
    Builds a multiple-choice prompt with a variable number of options.
    """
    letters = ["A", "B", "C", "D", "E", "F"]
    prompt = f"Pasartea: {context}\n\n"
    prompt += f"Galdera: {question}\n"
    for i, opt in enumerate(candidates):
        if opt is None or opt.strip() == "":
            continue
        prompt += f"{letters[i]}. {opt}\n"
    prompt += "Erantzuna:"
    return prompt

def build_fewshot_example(item):
    """Build a complete example with answer for few-shot context."""
    candidates = [o for o in item["candidates"] if o and o.strip()]
    letters = ["A", "B", "C", "D", "E", "F"]
    
    prompt = format_prompt_reading(
        context=item["context"],
        question=item["question"],
        candidates=candidates
    )
    answer_letter = letters[item["answer"]]
    return prompt + " " + answer_letter

def build_fewshot_context(items, current_idx, k=5, seed=42):
    """Build k-shot context excluding the current item."""
    rng = random.Random(seed + current_idx)
    # Pool excludes current example
    pool = [items[i] for i in range(len(items)) if i != current_idx]
    fewshot_examples = rng.sample(pool, min(k, len(pool)))
    texts = [build_fewshot_example(ex) for ex in fewshot_examples]
    return "\n\n".join(texts)

# ==================== TOKENIZATION ====================
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id

# Create cache directory if it doesn't exist
os.makedirs("cache", exist_ok=True)

cache_file = f"cache/eusreading_{MODEL_NAME}_fewshot_tokenized.jsonl"
queries_file = f"cache/eusreading_{MODEL_NAME}_queries.jsonl"

print(f"\nTokenizing dataset with new tokenizer (5-shot)...")
print(f"Cache file: {cache_file}")

with open(cache_file, "w") as f, \
     open(queries_file, "w") as f2:
    for idx, item in enumerate(tqdm(valid_items, desc=f"Tokenizing ({MODEL_NAME} few-shot)")):
        # Build 5-shot context
        fewshot_context = build_fewshot_context(valid_items, idx, k=5)
        
        # Build query (without answer)
        candidates = [o for o in item["candidates"] if o and o.strip()]
        query_text = format_prompt_reading(
            context=item["context"],
            question=item["question"],
            candidates=candidates
        )
        
        # Full prompt with few-shot examples
        prompt_text = fewshot_context + "\n\n" + query_text
        
        prompt_tokens = tokenizer.tokenize(prompt_text)
        query_tokens = tokenizer.tokenize(query_text) 
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        
        # Tokenize choice tokens (letters only)
        letters = ["A", "B", "C", "D", "E", "F"]
        choice_tokens = {}
        for i in range(len(candidates)):
            letter = letters[i]
            choice_tokens[letter] = tokenizer.tokenize(" " + letter)
        
        f.write(json.dumps({
            "id": idx,
            "prompt_tokens": prompt_tokens,
            "choice_tokens": choice_tokens,
            "num_candidates": len(candidates),
            "gold": item["answer"]
        }) + "\n")
        
        f2.write(json.dumps({
            "id": idx,
            "query_tokens": query_tokens,
            "query_ids": query_ids,
            "num_candidates": len(candidates),
            "gold": item["answer"]
        }) + "\n")

print(f"✓ Tokenization cached")

# ==================== EVALUATION ====================
correct = 0
total = 0

results_path = f"cache/eusreading_{MODEL_NAME}_eval_results.jsonl"

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

        # ----- Build full sequences for each choice -----
        full_ids = []
        letters = ["A", "B", "C", "D", "E", "F"]
        num_candidates = item["num_candidates"]
        
        for i in range(num_candidates):
            letter = letters[i]
            choice_text = tokenizer.convert_tokens_to_string(
                item["choice_tokens"][letter]
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
            "scores": scores.tolist(),
            "num_candidates": num_candidates
        }) + "\n")

accuracy = correct / total

print(f"\n{'='*60}")
print(f"Results for: {MODEL_NAME}")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
print(f"Results saved to: {results_path}")
print(f"{'='*60}\n")

# ==================== COMPARE WITH BASELINE ====================
baseline_results = "cache/eusreading_latxa_eval_results.jsonl"
if os.path.exists(baseline_results):
    print("\nComparing with baseline Latxa...")
    baseline_correct = 0
    baseline_total = 0
    
    with open(baseline_results) as f:
        for line in f:
            result = json.loads(line)
            if result["correct"]:
                baseline_correct += 1
            baseline_total += 1
    
    baseline_accuracy = baseline_correct / baseline_total
    improvement = accuracy - baseline_accuracy
    
    print(f"Baseline Latxa accuracy: {baseline_accuracy:.4f}")
    print(f"New tokenizer accuracy: {accuracy:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
else:
    print(f"\nBaseline results not found at {baseline_results}")
    print("Run the baseline evaluation first to compare.")