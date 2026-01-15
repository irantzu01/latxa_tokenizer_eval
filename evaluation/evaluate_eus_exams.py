#!/usr/bin/env python3
"""
Evaluate multiple models on EusExams dataset:
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
import datasets

# Add paths for dynamic tokenization
project_root = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization")
sys.path.append(project_root)

scripts_path = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/latxa_tokenizer_eval/scripts")
sys.path.append(scripts_path)

from evaluation_helper_functions import build_batch_tensors, score_choices
from tokenizations.dynamic_bpe import Dynamic_BPE
from scripts.dynamic_augmenter_new import DynamicAugmenter

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

# Load configs for EusExams
print("Loading the configs from json file...")
with open("eus_exams_configs_eu.json", "r") as f:
    configs = json.load(f)

CONFIGS = configs["configs"]

seed = 42
random.seed(seed)

answer2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
letters = ["A", "B", "C", "D"]

# ==================== DATASET LOADING ====================
def process_docs(dataset: datasets.Dataset):
    """Filter out examples with no answer."""
    def valid_example(example: dict) -> bool:
        """Check if an example is valid."""
        if example["answer"] not in [0, 1, 2, 3]:
            return False
        if example["candidates"] == ["", "", "", ""]:
            return False
        return True
    
    return dataset.filter(valid_example)

def load_eus_exams():
    """Load EusExams dataset with all configurations."""
    dataset = {}
    for config in CONFIGS:
        print(f"Loading config: {config}")
        dataset[config] = load_dataset("HiTZ/EusExams", name=config, split="test")
        dataset[config] = process_docs(dataset[config])
    return dataset

# ==================== HELPER FUNCTIONS ====================
def format_question(item):
    """Format a question for the model."""
    question = item["question"]
    candidates = item["candidates"]
    
    formatted_question = (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )
    return formatted_question

def build_fewshot_example(item):
    """Build a complete example with answer."""
    return format_question(item) + " " + answer2letter[item["answer"]]

def build_fewshot_context(dataset, current_idx, k=5):
    """Build k-shot context excluding current item."""
    current_item = dataset[current_idx]
    # Pool excludes current example
    pool = [dataset[i] for i in range(len(dataset)) if dataset[i] != current_item]
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
        for i in range(4):
            choice_text = " " + letters[i]
            choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        scores = score_choices(model, input_ids, attention_mask)
        
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
    print("Loading hypernet for dynamic tokenization...")
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
    
    print("Initializing dynamic augmenter...")
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
        for i in range(4):
            choice_text = " " + letters[i]
            choice_tokens = dynamic_tokenize_texts([choice_text], dynamic_bpe, max_merges=10)[0]
            choice_ids = augmenter.tokens_to_ids([choice_tokens])[0]
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        scores = score_choices(model, input_ids, attention_mask)
        
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
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per config")
    parser.add_argument("--config", type=str, default=None, 
                       help="Specific config to evaluate (otherwise all configs)")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['description']}")
    print(f"Model path: {model_config['path']}")
    print(f"Few-shot: {args.shots}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading EusExams dataset...")
    datasets = load_eus_exams()
    
    # Filter to specific config if requested
    if args.config:
        if args.config not in datasets:
            raise ValueError(f"Config {args.config} not found in available configs: {list(datasets.keys())}")
        datasets = {args.config: datasets[args.config]}
    
    print(f"Configs to evaluate: {list(datasets.keys())}")
    
    # Load model and tokenizer once
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("\nLoading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_config["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Evaluate each config
    all_results = {}
    for config_name, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating config: {config_name}")
        print(f"Dataset size: {len(dataset)}")
        print(f"{'='*60}\n")
        
        if args.limit:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
            print(f"Limited to {len(dataset)} examples")
        
        # Evaluate
        if model_config["use_dynamic"]:
            accuracy, results = evaluate_dynamic_model(
                model, tokenizer, dataset, shots=args.shots, device=device
            )
        else:
            accuracy, results = evaluate_static_model(
                model, tokenizer, dataset, shots=args.shots, device=device
            )
        
        all_results[config_name] = {
            "accuracy": accuracy,
            "correct": sum(r['correct'] for r in results),
            "total": len(results)
        }
        
        # Save results for this config
        results_file = f"results/eusexams_{config_name}_{args.model}_{args.shots}shot.jsonl"
        
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        print(f"\n✓ {config_name}: {accuracy:.4f} ({all_results[config_name]['correct']}/{all_results[config_name]['total']})")
        print(f"  Results saved to: {results_file}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {model_config['description']}")
    print(f"{'='*60}")
    
    total_correct = sum(r['correct'] for r in all_results.values())
    total_examples = sum(r['total'] for r in all_results.values())
    overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
    
    for config_name, stats in all_results.items():
        print(f"{config_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print(f"\nOverall: {overall_accuracy:.4f} ({total_correct}/{total_examples})")
    print(f"{'='*60}\n")
    
    # Save summary
    summary_file = f"results/eusexams_{args.model}_{args.shots}shot_summary.json"
    summary = {
        "model": args.model,
        "description": model_config['description'],
        "shots": args.shots,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_examples": total_examples,
        "configs": all_results
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()