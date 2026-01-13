#!/usr/bin/env python3
"""
Evaluate multiple models on EusTrivia dataset:
1. Original Latxa
2. Latxa with dynamic tokenization
3. Latxa with Basque tokenizer
4. Latxa with Basque tokenizer + FOCUS
"""

import sys
import os
import json
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from tqdm import tqdm

# Add paths for dynamic tokenization
project_root = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization")
sys.path.append(project_root)

scripts_path = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/latxa_tokenizer_eval/scripts")
sys.path.append(scripts_path)

from evaluation_helper_functions import build_batch_tensors, score_choices
from tokenizations.dynamic_bpe import Dynamic_BPE
from dynamic_augmenter_new import DynamicAugmenter

# ==================== CONFIGURATION ====================
MODELS = {
    "latxa_original": {
        "path": "HiTZ/latxa-7b-v1.2",
        "tokenizer_path": "HiTZ/latxa-7b-v1.2",
        "use_dynamic": False,
        "description": "Original Latxa 7B"
    },
    "latxa_dynamic": {
        "path": "HiTZ/latxa-7b-v1.2",
        "tokenizer_path": "HiTZ/latxa-7b-v1.2",
        "use_dynamic": True,
        "description": "Latxa 7B + Dynamic Tokenization"
    },
    "latxa_basque_tokenizer": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final"),
        "use_dynamic": False,
        "description": "Latxa 7B + Basque Tokenizer (250k)"
    },
    "latxa_basque_focus": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS/final"),
        "use_dynamic": False,
        "description": "Latxa 7B + Basque Tokenizer + FOCUS"
    },
}

seed = 42
random.seed(seed)

answer2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
letters = ["A", "B", "C", "D"]

# ==================== HELPER FUNCTIONS ====================
def format_question(doc) -> str:
    """Format a question for the model."""
    candidates = doc["candidates"]
    num_choices = len(candidates)
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Galdera: {doc['question']}\n{formatted_choices}\nErantzuna:"

def build_fewshot_example(doc):
    """Build a complete example with answer."""
    return format_question(doc) + " " + answer2letter[doc["answer"]]

def build_fewshot_context(dataset, current_idx, k=5):
    """Build k-shot context excluding current item."""
    # Pool excludes current example
    pool = [dataset[i] for i in range(len(dataset)) if i != current_idx]
    few_shot_examples = random.sample(pool, min(k, len(pool)))
    texts = [build_fewshot_example(ex) for ex in few_shot_examples]
    return "\n\n".join(texts)

def evaluate_static_model(model, tokenizer, dataset, shots=5, device='cuda'):
    """Evaluate a model with static tokenization."""
    correct = 0
    total = 0
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
        # Build prompt with few-shot examples
        fewshot_context = build_fewshot_context(dataset, idx, k=shots)
        query_text = format_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Build sequences for each choice
        full_ids = []
        num_choices = len(item["candidates"])
        for i in range(num_choices):
            choice_text = " " + letters[i]
            choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        
        # Score one at a time to avoid OOM
        scores_list = []
        for j in range(len(full_ids)):
            single_input = input_ids[j:j+1]
            single_mask = attention_mask[j:j+1]
            score = score_choices(model, single_input, single_mask)
            scores_list.append(score.item())
            torch.cuda.empty_cache()
        
        scores = torch.tensor(scores_list)
        pred = torch.argmax(scores).item()
        is_correct = (pred == item["answer"])
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "id": idx,
            "gold": item["answer"],
            "prediction": pred,
            "correct": is_correct,
            "scores": scores.tolist()
        })
    
    accuracy = correct / total
    return accuracy, results

def evaluate_dynamic_model(model, tokenizer, dataset, shots=5, device='cuda'):
    """Evaluate a model with dynamic tokenization."""
    # Initialize dynamic components
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
    
    augmenter = DynamicAugmenter(
        model=model,
        latxa_tokenizer=tokenizer,
        hypernet=hypernet,
        hypernet_tokenizer=hypernet_tokenizer,
        cache_limit=50000,
        device=device
    )
    
    correct = 0
    total = 0
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    # Import dynamic tokenization helper
    from evaluation_helper_functions import dynamic_tokenize_texts
    
    for idx, item in enumerate(tqdm(dataset, desc="Evaluating (dynamic)")):
        # Build prompt with few-shot examples
        fewshot_context = build_fewshot_context(dataset, idx, k=shots)
        query_text = format_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize with dynamic BPE
        prompt_tokens = dynamic_tokenize_texts([prompt_text], dynamic_bpe, max_merges=10)[0]
        prompt_ids = augmenter.tokens_to_ids([prompt_tokens])[0]
        
        # Build sequences for each choice
        full_ids = []
        num_choices = len(item["candidates"])
        for i in range(num_choices):
            choice_text = " " + letters[i]
            choice_tokens = dynamic_tokenize_texts([choice_text], dynamic_bpe, max_merges=10)[0]
            choice_ids = augmenter.tokens_to_ids([choice_tokens])[0]
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices one at a time
        scores_list = []
        for seq_ids in full_ids:
            input_ids_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            score = score_choices(model, input_ids_tensor, attention_mask_tensor)
            scores_list.append(score.item())
            torch.cuda.empty_cache()
        
        scores = torch.tensor(scores_list)
        pred = torch.argmax(scores).item()
        is_correct = (pred == item["answer"])
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "id": idx,
            "gold": item["answer"],
            "prediction": pred,
            "correct": is_correct,
            "scores": scores.tolist()
        })
    
    accuracy = correct / total
    return accuracy, results

# ==================== MAIN EVALUATION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODELS.keys()),
                       help="Model to evaluate")
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['description']}")
    print(f"Model path: {model_config['path']}")
    print(f"Few-shot: {args.shots}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading EusTrivia dataset...")
    dataset = load_dataset("HiTZ/EusTrivia", name="default", split="test")
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limited to {args.limit} examples")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_config["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Evaluate
    if model_config["use_dynamic"]:
        accuracy, results = evaluate_dynamic_model(
            model, tokenizer, dataset, shots=args.shots, device=device
        )
    else:
        accuracy, results = evaluate_static_model(
            model, tokenizer, dataset, shots=args.shots, device=device
        )
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_file = f"results/eustrivia_{args.model}_{args.shots}shot.jsonl"
    
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {model_config['description']}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()