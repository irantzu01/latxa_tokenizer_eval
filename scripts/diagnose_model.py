#!/usr/bin/env python3
"""
Diagnose why the 250K model is performing poorly on EusProficiency
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
import json

# Paths
original_model = "HiTZ/latxa-7b-v1.2"
model_250k = os.path.expanduser("~/tmp/models/latxa7b_basque_aligned_250k/final")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("DIAGNOSTIC REPORT: 250K Realigned Model")
print("="*70)

# ==================== 1. TOKENIZER COMPARISON ====================
print("\n1. TOKENIZER COMPARISON")
print("-"*70)

tok_original = AutoTokenizer.from_pretrained(original_model)
tok_250k = AutoTokenizer.from_pretrained(model_250k)

print(f"Original Latxa vocab size: {len(tok_original)}")
print(f"250K model vocab size:     {len(tok_250k)}")

if len(tok_250k) == len(tok_original):
    print("⚠️  WARNING: Vocab sizes are IDENTICAL - new tokenizer may not have been applied!")
else:
    print(f"✓ Vocab size changed by {len(tok_250k) - len(tok_original)} tokens")

# Test tokenization differences
test_sentences = [
    "Euskal Herria da gure herria",
    "Gaur egun teknologia",
    "Bilbo hiriak"
]

print("\nTokenization comparison:")
for sent in test_sentences:
    tok_orig = tok_original.tokenize(sent)
    tok_new = tok_250k.tokenize(sent)
    
    print(f"\n  '{sent}':")
    print(f"    Original ({len(tok_orig):2d} tokens): {' | '.join(tok_orig)}")
    print(f"    New      ({len(tok_new):2d} tokens): {' | '.join(tok_new)}")
    
    if tok_orig == tok_new:
        print("    ⚠️  IDENTICAL tokenization!")

# ==================== 2. MODEL COMPARISON ====================
print("\n" + "="*70)
print("2. MODEL WEIGHT COMPARISON")
print("-"*70)

print("Loading models...")
model_orig = AutoModelForCausalLM.from_pretrained(original_model, device_map="auto", torch_dtype=torch.bfloat16)
model_250k_loaded = AutoModelForCausalLM.from_pretrained(model_250k, device_map="auto", torch_dtype=torch.bfloat16)

# Compare embeddings
embed_orig = model_orig.get_input_embeddings().weight
embed_250k = model_250k_loaded.get_input_embeddings().weight

print(f"\nEmbedding shapes:")
print(f"  Original: {embed_orig.shape}")
print(f"  250K:     {embed_250k.shape}")

# Compare overlapping tokens
min_vocab = min(embed_orig.shape[0], embed_250k.shape[0])

with torch.no_grad():
    # Sample 100 random tokens for comparison
    sample_indices = torch.randint(0, min_vocab, (100,))
    
    embed_orig_sample = embed_orig[sample_indices].cpu().float()
    embed_250k_sample = embed_250k[sample_indices].cpu().float()
    
    diff = (embed_orig_sample - embed_250k_sample).abs().mean().item()
    
    print(f"\nMean absolute difference (100 random tokens): {diff:.6f}")
    
    if diff < 0.0001:
        print("⚠️  CRITICAL: Embeddings are IDENTICAL to original!")
        print("   Your training did not modify the embeddings at all.")
    elif diff < 0.01:
        print("⚠️  WARNING: Embeddings changed very little")
    else:
        print(f"✓ Embeddings have changed (good)")

# ==================== 3. GENERATION TEST ====================
print("\n" + "="*70)
print("3. GENERATION QUALITY TEST")
print("-"*70)

model_orig.eval()
model_250k_loaded.eval()

test_prompts = [
    "Euskal Herria da",
    "Gaur egun, teknologia",
    "Bilbo hiriak"
]

print("\nComparing generation quality:\n")

for prompt in test_prompts:
    print(f"Prompt: '{prompt}'")
    
    # Original model
    inputs_orig = tok_original(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_orig = model_orig.generate(
            **inputs_orig,
            max_new_tokens=20,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tok_original.eos_token_id
        )
    generated_orig = tok_original.decode(outputs_orig[0], skip_special_tokens=True)
    
    # 250K model
    inputs_250k = tok_250k(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs_250k = model_250k_loaded.generate(
            **inputs_250k,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tok_250k.eos_token_id
        )
    generated_250k = tok_250k.decode(outputs_250k[0], skip_special_tokens=True)
    
    print(f"  Original: {generated_orig}")
    print(f"  250K:     {generated_250k}")
    
    if generated_orig == generated_250k:
        print("  ⚠️  Generations are IDENTICAL!")
    print()

# ==================== 4. EVALUATION SAMPLE ANALYSIS ====================
print("="*70)
print("4. ANALYZING EVALUATION PREDICTIONS")
print("-"*70)

results_file = "cache/eusproficiency_250k_improved_eval_results.jsonl"

if os.path.exists(results_file):
    print(f"\nAnalyzing {results_file}...\n")
    
    predictions_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    gold_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    score_analysis = []
    
    with open(results_file) as f:
        for line in f:
            result = json.loads(line)
            predictions_dist[result['prediction']] += 1
            gold_dist[result['gold']] += 1
            score_analysis.append(result['scores'])
    
    print("Prediction distribution:")
    for choice, count in predictions_dist.items():
        choice_letter = ['A', 'B', 'C', 'D'][choice]
        print(f"  {choice_letter}: {count:4d} ({count/sum(predictions_dist.values())*100:.1f}%)")
    
    print("\nGold distribution:")
    for choice, count in gold_dist.items():
        choice_letter = ['A', 'B', 'C', 'D'][choice]
        print(f"  {choice_letter}: {count:4d} ({count/sum(gold_dist.values())*100:.1f}%)")
    
    # Check if model is just guessing randomly or has a bias
    import numpy as np
    pred_entropy = -sum([(p/sum(predictions_dist.values())) * np.log2(p/sum(predictions_dist.values())) 
                         for p in predictions_dist.values() if p > 0])
    
    print(f"\nPrediction entropy: {pred_entropy:.3f} (max=2.0 for uniform)")
    
    if pred_entropy > 1.9:
        print("⚠️  Model appears to be guessing randomly!")
    
    # Analyze score distributions
    scores_array = np.array(score_analysis)
    print(f"\nScore statistics:")
    print(f"  Mean:   {scores_array.mean():.4f}")
    print(f"  Std:    {scores_array.std():.4f}")
    print(f"  Min:    {scores_array.min():.4f}")
    print(f"  Max:    {scores_array.max():.4f}")
    
    # Check if scores are all very similar (indicating the model isn't confident)
    score_ranges = scores_array.max(axis=1) - scores_array.min(axis=1)
    print(f"\nScore range per question:")
    print(f"  Mean range: {score_ranges.mean():.4f}")
    print(f"  Median range: {np.median(score_ranges):.4f}")
    
    if score_ranges.mean() < 0.1:
        print("⚠️  Score ranges are very small - model is not confident!")
else:
    print(f"\n⚠️  Results file not found: {results_file}")
    print("Run the evaluation first.")

# ==================== 5. RECOMMENDATIONS ====================
print("\n" + "="*70)
print("5. RECOMMENDATIONS")
print("="*70)

recommendations = []

# Check vocab
if len(tok_250k) == len(tok_original):
    recommendations.append("⚠️  Your tokenizer is identical to the original - the new tokenizer wasn't properly applied")

# Check embeddings
if diff < 0.01:
    recommendations.append("⚠️  Embeddings barely changed - increase learning rate or train longer")

# Check if evaluation exists
if not os.path.exists(results_file):
    recommendations.append("📋 Run the evaluation script first to generate results")

if recommendations:
    print("\nIssues found:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("\n✓ No obvious issues found in tokenizer or model")
    print("  The problem may be in the evaluation script or task mismatch")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. If tokenizer is identical to original:
   - Check that you're using the correct tokenizer_dir in training
   - Verify basque_tokenizer_hf was created correctly

2. If embeddings barely changed:
   - Retrain with higher learning rate (1e-3 instead of 5e-4)
   - Train for more epochs
   - Use more data (500K or 1M samples)

3. If model generates poorly:
   - Check that embeddings are trainable (requires_grad=True)
   - Verify training loss actually decreased
   - Test with the best checkpoint (lowest perplexity)

4. Compare with baseline:
   - Run evaluation on original Latxa-7B
   - Check if the task itself is hard (baseline might also be ~25%)
""")

print("="*70)