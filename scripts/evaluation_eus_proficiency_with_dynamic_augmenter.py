import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

# Add path to your scripts directory
scripts_path = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/latxa_tokenizer_eval/scripts"
)
sys.path.append(scripts_path)

from evaluation_helper_functions import (
    dynamic_tokenize_texts,
    build_batch_tensors,
    score_choices
)

from tokenizations.dynamic_bpe import Dynamic_BPE
from dynamic_augmenter_new import DynamicAugmenter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from datasets import load_dataset
from tqdm import tqdm
import random

# ==================== CONFIGURATION ====================
MODEL_NAME = "latxa_dynamic"

print(f"\n{'='*60}")
print(f"Evaluating ORIGINAL Latxa with dynamic tokenization")
print(f"{'='*60}\n")

# ==================== LOAD DATASET ====================
ds = load_dataset("HiTZ/EusProficiency", split="test")

# ==================== LOAD ORIGINAL LATXA MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading ORIGINAL Latxa model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
model.to(device)
model.eval()
print(f"✓ Original Latxa model loaded")
print(model.get_input_embeddings().weight.shape[1])

print("Loading hypernet for dynamic tokenization...")
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental",
    trust_remote_code=True
)
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)
print("✓ Hypernet loaded")

# ==================== INITIALIZE DYNAMIC TOKENIZER ====================
print("Initializing dynamic BPE tokenizer...")
dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)
print("✓ Dynamic BPE initialized")

# ==================== INITIALIZE DYNAMIC AUGMENTER ====================
print("Initializing dynamic augmenter with projection adapter...")
augmenter = DynamicAugmenter(
    model=model,
    latxa_tokenizer=latxa_tokenizer,  # Using ORIGINAL Latxa tokenizer
    hypernet=hypernet,
    hypernet_tokenizer=hypernet_tokenizer,
    cache_limit=50000,
    device=device
)
print("✓ Dynamic augmenter initialized")

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

# ==================== TOKENIZATION WITH DYNAMIC BPE ====================
CHOICES = [" A", " B", " C", " D"]

# Determine pad_id from ORIGINAL Latxa tokenizer
pad_id = latxa_tokenizer.pad_token_id
if pad_id is None:
    pad_id = latxa_tokenizer.eos_token_id

# Create cache directory
os.makedirs("cache", exist_ok=True)

cache_file = f"cache/eusproficiency_{MODEL_NAME}_fewshot_tokenized.jsonl"

print(f"\nTokenizing dataset with dynamic BPE...")
print(f"Cache file: {cache_file}")

with open(cache_file, "w") as f:
    for idx, item in enumerate(tqdm(ds, desc=f"Tokenizing ({MODEL_NAME})")):
        fewshot_context = build_fewshot_context(ds, idx, k=5)
        query_text = build_doc_text(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize with dynamic BPE using the helper function
        prompt_tokens = dynamic_tokenize_texts([prompt_text], dynamic_bpe, max_merges=10)[0]
        
        # Tokenize choices with dynamic BPE
        choice_tokens = {}
        for c in ["A", "B", "C", "D"]:
            choice_text = " " + c
            choice_tokens[c] = dynamic_tokenize_texts([choice_text], dynamic_bpe, max_merges=10)[0]
        
        f.write(json.dumps({
            "id": idx,
            "prompt_tokens": prompt_tokens,
            "choice_tokens": choice_tokens,
            "gold": item["answer"]
        }) + "\n")

print("✓ Dynamic tokenization cached")

# ==================== EVALUATION ====================
correct = 0
total = 0

results_path = f"cache/eusproficiency_{MODEL_NAME}_eval_results.jsonl"

print(f"\nEvaluating model with dynamic augmentation...")
print(f"Results will be saved to: {results_path}")

with open(cache_file) as fin, \
     open(results_path, "w") as fout:

    for line in tqdm(fin, desc=f"Evaluating ({MODEL_NAME})"):

        item = json.loads(line)

        # ----- Convert dynamic tokens to IDs using augmenter -----
        prompt_tokens = item["prompt_tokens"]
        
        # Convert prompt tokens to IDs
        prompt_ids_list = augmenter.tokens_to_ids([prompt_tokens])
        prompt_ids = prompt_ids_list[0]
        
        # ----- Build full sequences for each choice -----
        full_ids = []
        for c in ["A", "B", "C", "D"]:
            choice_tokens = item["choice_tokens"][c]
            
            # Convert choice tokens to IDs
            choice_ids_list = augmenter.tokens_to_ids([choice_tokens])
            choice_ids = choice_ids_list[0]
            
            # Combine prompt + choice
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
print(f"Base vocab size: {augmenter.base_vocab_size}")
print(f"Current vocab size: {augmenter.current_vocab_size}")
print(f"Dynamic tokens cached: {len(augmenter.cache)}")
print(f"Results saved to: {results_path}")
print(f"{'='*60}\n")

# ==================== COMPARE WITH BASELINE ====================
baseline_results = "cache/eusproficiency_latxa_eval_results.jsonl"
if os.path.exists(baseline_results):
    print("\nComparing with baseline Latxa (static tokenization)...")
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
    
    print(f"Latxa (static) accuracy: {baseline_accuracy:.4f}")
    print(f"Latxa (dynamic) accuracy: {accuracy:.4f}")
    print(f"Dynamic improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
else:
    print(f"\nBaseline results not found at {baseline_results}")
    print("Run the baseline evaluation first to compare.")

# ==================== SAVE ADAPTER ====================
print("\nSaving trained adapter...")
augmenter.save_adapter(f"models/{MODEL_NAME}_adapter.pt")
print("✓ Adapter saved for future use")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"Model: Original Latxa 7B")
print(f"Tokenization: Dynamic BPE")
print(f"Adapter: Latxa (8192) <-> Llama3 (4096)")
print(f"Dynamic tokens added: {len(augmenter.cache)}")
print(f"Final accuracy: {accuracy:.4f}")
print("="*60)
