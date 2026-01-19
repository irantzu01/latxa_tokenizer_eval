#!/usr/bin/env python3
"""
Test if models can generate coherent text.
This will help diagnose if the model is fundamentally broken.
"""

import sys
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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
}

# Test prompts in Basque
TEST_PROMPTS = [
    "Euskal Herria",
    "Gaur egun",
    "Nik uste dut",
    "Galdera: Nola zaude?\nErantzuna:",
    "Bilbo",
]

def test_generation(model, tokenizer, prompt, name, max_new_tokens=50, temperature=0.7):
    """Generate text from a prompt and analyze the output."""
    print(f"\n{'='*80}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*80}")
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    print(f"\nInput tokenization ({len(input_tokens)} tokens):")
    print(f"  Tokens: {' '.join(input_tokens)}")
    print(f"  Token IDs: {input_ids[0].tolist()}")
    
    # Generate with different strategies
    print(f"\n--- GREEDY DECODING (temperature=0) ---")
    with torch.no_grad():
        greedy_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    greedy_text = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    print(f"Generated: {greedy_text}")
    
    # Check if it's just repeating
    generated_only = tokenizer.decode(greedy_output[0][len(input_ids[0]):], skip_special_tokens=True)
    print(f"New tokens only: '{generated_only}'")
    
    # Count unique tokens in generation
    new_tokens = greedy_output[0][len(input_ids[0]):].tolist()
    unique_tokens = len(set(new_tokens))
    print(f"Generated {len(new_tokens)} tokens, {unique_tokens} unique")
    
    # Check for repetition
    if len(new_tokens) > 0:
        most_common_token = max(set(new_tokens), key=new_tokens.count)
        most_common_count = new_tokens.count(most_common_token)
        repetition_ratio = most_common_count / len(new_tokens)
        
        most_common_text = tokenizer.decode([most_common_token])
        print(f"Most repeated token: '{most_common_text}' ({most_common_count}/{len(new_tokens)} = {repetition_ratio:.1%})")
        
        if repetition_ratio > 0.5:
            print("⚠️  WARNING: High repetition detected!")
    
    print(f"\n--- SAMPLING (temperature={temperature}) ---")
    with torch.no_grad():
        sampled_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    sampled_text = tokenizer.decode(sampled_output[0], skip_special_tokens=True)
    print(f"Generated: {sampled_text}")
    
    # Analyze output tokens
    output_tokens = tokenizer.convert_ids_to_tokens(sampled_output[0])
    new_output_tokens = output_tokens[len(input_tokens):]
    print(f"New tokens: {' '.join(new_output_tokens[:20])}{'...' if len(new_output_tokens) > 20 else ''}")

def test_next_token_probabilities(model, tokenizer, prompt, top_k=20):
    """Check what tokens the model predicts next with highest probability."""
    print(f"\n{'='*80}")
    print(f"Next Token Probabilities for: '{prompt}'")
    print(f"{'='*80}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # Last token's predictions
        probs = torch.softmax(logits, dim=-1)
    
    # Get top k predictions
    top_probs, top_ids = torch.topk(probs, top_k)
    
    print(f"\nTop {top_k} most likely next tokens:")
    for i, (prob, token_id) in enumerate(zip(top_probs, top_ids), 1):
        token = tokenizer.decode([token_id])
        print(f"  {i:2d}. '{token}' (ID: {token_id.item()}) - {prob.item():.4f} ({prob.item()*100:.2f}%)")
    
    # Check if top prediction is reasonable
    top_token = tokenizer.decode([top_ids[0]])
    top_prob = top_probs[0].item()
    
    if top_prob > 0.9:
        print(f"\n⚠️  WARNING: Very high probability ({top_prob:.1%}) for single token '{top_token}'")
    
    # Check for suspicious patterns
    top_10_tokens = [tokenizer.decode([tid]) for tid in top_ids[:10]]
    if top_10_tokens.count(top_10_tokens[0]) > 5:
        print(f"⚠️  WARNING: Top predictions show repetition!")

def compare_models_side_by_side(models_to_test, prompt, max_new_tokens=30):
    """Generate from all models side by side for comparison."""
    print(f"\n{'#'*80}")
    print(f"SIDE-BY-SIDE COMPARISON")
    print(f"Prompt: '{prompt}'")
    print(f"{'#'*80}")
    
    results = {}
    
    for model_name in models_to_test:
        model_config = MODELS[model_name]
        
        print(f"\nLoading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_config["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        results[model_name] = generated_text
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    for model_name, text in results.items():
        print(f"\n{model_name}:")
        print(f"  {text}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="latxa_basque_tokenizer",
                       choices=list(MODELS.keys()),
                       help="Model to test")
    parser.add_argument("--all-models", action="store_true",
                       help="Test all models")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all models side-by-side")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Custom prompt to test")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if args.compare:
        # Compare all models
        models_to_test = list(MODELS.keys())
        test_prompts = [args.prompt] if args.prompt else TEST_PROMPTS[:3]
        
        for prompt in test_prompts:
            compare_models_side_by_side(models_to_test, prompt, args.max_tokens)
        return
    
    # Test individual model(s)
    models_to_test = list(MODELS.keys()) if args.all_models else [args.model]
    prompts_to_test = [args.prompt] if args.prompt else TEST_PROMPTS
    
    for model_name in models_to_test:
        model_config = MODELS[model_name]
        
        print(f"\n{'#'*80}")
        print(f"# TESTING MODEL: {model_name}")
        print(f"{'#'*80}")
        
        # Load model
        print("\nLoading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(model_config["path"])
        tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
        model.to(device)
        model.eval()
        print("✓ Loaded")
        
        # Test generation
        for prompt in prompts_to_test:
            test_generation(model, tokenizer, prompt, model_name, args.max_tokens)
            test_next_token_probabilities(model, tokenizer, prompt)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()