#!/usr/bin/env python3
"""
Debug script to investigate why models show extreme bias toward certain answer choices.
Analyzes tokenization and scoring for a single example across different tokenizers.
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Add paths
project_root = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization")
sys.path.append(project_root)

scripts_path = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/latxa_tokenizer_eval/scripts")
sys.path.append(scripts_path)

from evaluation_helper_functions import build_batch_tensors, score_choices

# ==================== CONFIGURATION ====================
MODELS = {
    "latxa_original": {
        "path": "HiTZ/latxa-7b-v1.2",
        "tokenizer_path": "HiTZ/latxa-7b-v1.2",
    },
    "latxa_basque_tokenizer": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final"),
    },
    "latxa_basque_focus": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS/final"),
    },
    "latxa_basque_tokenizer_improved": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_improved/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_improved/final"),
    },
    "latxa_basque_focus_improved": {
        "path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved/final"),
        "tokenizer_path": os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved/final"),
    },
}

letters = ["A", "B", "C", "D"]

def format_eusproficiency_question(item):
    """Format EusProficiency question."""
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

def debug_tokenization(text, tokenizer, name):
    """Debug how a text is tokenized."""
    print(f"\n{'='*80}")
    print(f"TOKENIZER: {name}")
    print(f"{'='*80}")
    
    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    print(f"Text: '{text}'")
    print(f"Number of tokens: {len(token_ids)}")
    print(f"\nToken IDs: {token_ids}")
    print(f"\nTokens: {tokens}")
    
    # Decode back
    decoded = tokenizer.decode(token_ids)
    print(f"\nDecoded: '{decoded}'")
    
    return token_ids, tokens

def debug_answer_choices(tokenizer, name):
    """Debug how answer choices are tokenized."""
    print(f"\n{'='*80}")
    print(f"ANSWER CHOICE TOKENIZATION: {name}")
    print(f"{'='*80}")
    
    for letter in letters:
        choice_text = " " + letter
        token_ids = tokenizer.encode(choice_text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        print(f"\nChoice: '{choice_text}'")
        print(f"  Token IDs: {token_ids}")
        print(f"  Tokens: {tokens}")
        print(f"  Num tokens: {len(token_ids)}")

def debug_full_scoring(model, tokenizer, item, name, device='cuda'):
    """Debug the full scoring process for one example."""
    print(f"\n{'='*80}")
    print(f"FULL SCORING DEBUG: {name}")
    print(f"{'='*80}")
    
    # Format question
    query_text = format_eusproficiency_question(item)
    print(f"\nQuestion text (last 200 chars):\n...{query_text[-200:]}")
    
    # Tokenize prompt
    prompt_ids = tokenizer.encode(query_text, add_special_tokens=False)
    print(f"\nPrompt tokens: {len(prompt_ids)}")
    print(f"Last 10 prompt tokens: {tokenizer.convert_ids_to_tokens(prompt_ids[-10:])}")
    
    # Build sequences for each choice
    print(f"\n{'='*60}")
    print("BUILDING SEQUENCES FOR EACH CHOICE")
    print(f"{'='*60}")
    
    full_ids = []
    for i, letter in enumerate(letters):
        choice_text = " " + letter
        choice_ids = tokenizer.encode(choice_text, add_special_tokens=False)
        full_seq = prompt_ids + choice_ids
        full_ids.append(full_seq)
        
        print(f"\nChoice {letter}:")
        print(f"  Choice text: '{choice_text}'")
        print(f"  Choice token IDs: {choice_ids}")
        print(f"  Choice tokens: {tokenizer.convert_ids_to_tokens(choice_ids)}")
        print(f"  Full sequence length: {len(full_seq)}")
        print(f"  Last 5 tokens of full sequence: {tokenizer.convert_ids_to_tokens(full_seq[-5:])}")
    
    # Score choices
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    print(f"\n{'='*60}")
    print("SCORING")
    print(f"{'='*60}")
    print(f"Pad token ID: {pad_id}")
    
    # Score each choice individually
    print("\nScoring each choice individually:")
    individual_scores = []
    for i, (letter, seq_ids) in enumerate(zip(letters, full_ids)):
        input_ids = torch.tensor([seq_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            score = score_choices(model, input_ids, attention_mask)
        
        individual_scores.append(score.item())
        print(f"  {letter}: {score.item():.4f}")
    
    # Score as batch
    print("\nScoring as batch:")
    input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch attention_mask shape: {attention_mask.shape}")
    
    with torch.no_grad():
        batch_scores = score_choices(model, input_ids, attention_mask)
    
    for i, (letter, score) in enumerate(zip(letters, batch_scores)):
        print(f"  {letter}: {score.item():.4f}")
    
    # Prediction
    pred_idx = torch.argmax(batch_scores).item()
    gold_idx = item["answer"]
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Gold answer: {letters[gold_idx]} (index {gold_idx})")
    print(f"Predicted answer: {letters[pred_idx]} (index {pred_idx})")
    print(f"Correct: {pred_idx == gold_idx}")
    
    # Check if individual vs batch scores differ
    print(f"\n{'='*60}")
    print("SCORE COMPARISON (Individual vs Batch)")
    print(f"{'='*60}")
    for letter, ind_score, batch_score in zip(letters, individual_scores, batch_scores.tolist()):
        diff = abs(ind_score - batch_score)
        print(f"{letter}: Individual={ind_score:.4f}, Batch={batch_score:.4f}, Diff={diff:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="latxa_original",
                       choices=list(MODELS.keys()),
                       help="Model to debug")
    parser.add_argument("--example-id", type=int, default=0,
                       help="Example ID to debug (default: 0)")
    parser.add_argument("--all-models", action="store_true",
                       help="Debug all models")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print("\nLoading EusProficiency dataset...")
    dataset = load_dataset("HiTZ/EusProficiency", name="default", split="test")
    item = dataset[args.example_id]
    
    print(f"\n{'='*80}")
    print(f"EXAMPLE {args.example_id}")
    print(f"{'='*80}")
    print(f"Question: {item['question']}")
    print(f"Candidates:")
    for i, cand in enumerate(item['candidates']):
        marker = " ← GOLD" if i == item['answer'] else ""
        print(f"  {letters[i]}: {cand}{marker}")
    print(f"Gold answer: {letters[item['answer']]} (index {item['answer']})")
    
    # Determine which models to debug
    models_to_debug = list(MODELS.keys()) if args.all_models else [args.model]
    
    for model_name in models_to_debug:
        model_config = MODELS[model_name]
        
        print(f"\n\n{'#'*80}")
        print(f"# DEBUGGING MODEL: {model_name}")
        print(f"{'#'*80}")
        
        # Load model and tokenizer
        print("\nLoading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_config["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
        model.to(device)
        model.eval()
        print("✓ Loaded")
        
        # Debug answer choices tokenization
        debug_answer_choices(tokenizer, model_name)
        
        # Debug a simple prompt
        simple_prompt = "Erantzuna:"
        debug_tokenization(simple_prompt, tokenizer, model_name)
        
        # Debug full scoring
        debug_full_scoring(model, tokenizer, item, model_name, device)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        if not args.all_models:
            break
    
    print(f"\n{'='*80}")
    print("DEBUG COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()