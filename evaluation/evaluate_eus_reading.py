#!/usr/bin/env python3
"""
Evaluate models on EusReading dataset.
IMPORTANT: EusReading has passages with multiple questions per passage.
Each question should be evaluated with its corresponding passage context.
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
from collections import defaultdict

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

# ==================== DATASET PROCESSING ====================
def group_by_context(dataset):
    """
    Group questions by their context/passage.
    Returns: dict mapping context -> list of questions
    """
    context_groups = defaultdict(list)
    
    for idx, item in enumerate(dataset):
        context = item["context"]
        context_groups[context].append({
            "original_idx": idx,
            "question": item["question"],
            "candidates": item["candidates"],
            "answer": item["answer"],
            "context": context
        })
    
    return context_groups

def format_question(doc) -> str:
    """
    Format a single question with its candidates.
    Does NOT include the passage/context.
    """
    candidates = doc["candidates"]
    # Filter out empty candidates
    candidates = [c for c in candidates if c and c.strip()]
    num_choices = len(candidates)
    
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Galdera: {doc['question']}\n{formatted_choices}\nErantzuna:"

def format_passage_with_question(doc) -> str:
    """
    Format passage + question together.
    This is what we actually evaluate on.
    """
    candidates = doc["candidates"]
    candidates = [c for c in candidates if c and c.strip()]
    num_choices = len(candidates)
    
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Pasartea: {doc['context']}\n\nGaldera: {doc['question']}\n{formatted_choices}\nErantzuna:"

def build_fewshot_example(doc):
    """Build a complete few-shot example: passage + question + answer."""
    return format_passage_with_question(doc) + " " + letters[doc["answer"]]

def build_fewshot_context(dataset_items, current_item_idx, k=5):
    """
    Build k-shot context by sampling k random (passage, question) pairs.
    This matches the original evaluation: each example is passage + question + answer.
    
    Args:
        dataset_items: List of all dataset items (each has context, question, candidates, answer)
        current_item_idx: Index of current item to exclude
        k: Number of few-shot examples
    """
    # Pool excludes current example
    pool = [dataset_items[i] for i in range(len(dataset_items)) if i != current_item_idx]
    
    if len(pool) < k:
        k = len(pool)
    
    # Sample k random examples (each is a passage + question pair)
    few_shot_examples = random.sample(pool, k)
    
    # Build few-shot text
    texts = [build_fewshot_example(ex) for ex in few_shot_examples]
    return "\n\n".join(texts)

# ==================== EVALUATION FUNCTIONS ====================
def evaluate_static_model(model, tokenizer, dataset, shots=5, device='cuda'):
    """Evaluate a model with static tokenization."""
    correct = 0
    total = 0
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    # Convert dataset to list for easier indexing
    dataset_items = list(dataset)
    
    for idx, item in enumerate(tqdm(dataset_items, desc="Evaluating")):
        # Filter candidates
        candidates = [c for c in item["candidates"] if c and c.strip()]
        if len(candidates) < 2:
            continue
        
        # Build few-shot context: k random (passage, question) pairs
        fewshot_context = build_fewshot_context(dataset_items, idx, k=shots)
        
        # Current question with its passage
        query_text = format_passage_with_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        # Build sequences for each choice
        full_ids = []
        num_choices = len(candidates)
        for i in range(num_choices):
            choice_text = " " + letters[i]
            choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices ONE AT A TIME to avoid OOM
        scores_list = []
        for seq_ids in full_ids:
            input_ids_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            
            with torch.no_grad():
                score = score_choices(model, input_ids_tensor, attention_mask_tensor)
            scores_list.append(score.item())
            
            # Clear GPU cache after each choice
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
            "scores": scores.tolist(),
            "num_candidates": num_choices
        })
    
    accuracy = correct / total if total > 0 else 0
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
    
    from evaluation_helper_functions import dynamic_tokenize_texts
    
    # Convert dataset to list for easier indexing
    dataset_items = list(dataset)
    
    for idx, item in enumerate(tqdm(dataset_items, desc="Evaluating (dynamic)")):
        # Filter candidates
        candidates = [c for c in item["candidates"] if c and c.strip()]
        if len(candidates) < 2:
            continue
        
        # Build few-shot context: k random (passage, question) pairs
        fewshot_context = build_fewshot_context(dataset_items, idx, k=shots)
        
        # Current question with its passage
        query_text = format_passage_with_question(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        # Tokenize with dynamic BPE
        prompt_tokens = dynamic_tokenize_texts([prompt_text], dynamic_bpe, max_merges=10)[0]
        prompt_ids = augmenter.tokens_to_ids([prompt_tokens])[0]
        
        # Build sequences for each choice
        full_ids = []
        num_choices = len(candidates)
        for i in range(num_choices):
            choice_text = " " + letters[i]
            choice_tokens = dynamic_tokenize_texts([choice_text], dynamic_bpe, max_merges=10)[0]
            choice_ids = augmenter.tokens_to_ids([choice_tokens])[0]
            full_ids.append(prompt_ids + choice_ids)
        
        # Score choices ONE AT A TIME to avoid OOM
        scores_list = []
        for seq_ids in full_ids:
            input_ids_tensor = torch.tensor([seq_ids], dtype=torch.long, device=device)
            attention_mask_tensor = torch.ones_like(input_ids_tensor)
            
            with torch.no_grad():
                score = score_choices(model, input_ids_tensor, attention_mask_tensor)
            scores_list.append(score.item())
            
            # Clear GPU cache after each choice
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
            "scores": scores.tolist(),
            "num_candidates": num_choices
        })
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

# ==================== MAIN EVALUATION ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODELS.keys()),
                       help="Model to evaluate")
    parser.add_argument("--shots", type=int, default=5, 
                       help="Number of few-shot examples (passages)")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of questions")
    args = parser.parse_args()
    
    model_config = MODELS[args.model]
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['description']}")
    print(f"Model path: {model_config['path']}")
    print(f"Few-shot: {args.shots} passages")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading EusReading dataset...")
    dataset = load_dataset("HiTZ/EusReading", name="default", split="test")
    
    print(f"Total dataset size: {len(dataset)} questions")
    
    if args.limit:
        dataset = dataset.select(range(args.limit))
        print(f"Limited to {args.limit} questions")
    
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
    results_file = f"results/eusreading_{args.model}_{args.shots}shot.jsonl"
    
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