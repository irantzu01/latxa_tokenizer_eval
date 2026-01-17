#!/usr/bin/env python3
"""
Compare results and tokenization between two models.
Analyze cases where predictions differ and examine tokenization differences.
"""

import sys
import os
import json
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict

# Add paths for dynamic tokenization
project_root = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization")
sys.path.append(project_root)

scripts_path = os.path.expanduser("~/MASTER/WiSe25/Lab Rotation/latxa_tokenizer_eval/scripts")
sys.path.append(scripts_path)

from tokenizations.dynamic_bpe import Dynamic_BPE

# ==================== DATASET FORMATTERS ====================
def format_eusreading_question(item):
    """Format EusReading question."""
    question = item["question"]
    candidates = item["candidates"]
    
    # Handle variable number of candidates
    letters = ["A", "B", "C", "D"]
    formatted_parts = [f"Galdera: {question}"]
    
    for i, candidate in enumerate(candidates):
        if i < len(letters):
            formatted_parts.append(f"{letters[i]}: {candidate}")
    
    formatted_parts.append("Erantzuna:")
    formatted_question = "\n".join(formatted_parts)
    
    return formatted_question

def format_eusproficiency_question(item):
    """Format EusProficiency question."""
    return format_eusreading_question(item)  # Same format

def format_belebele_question(item):
    """Format Belebele question."""
    flores_passage = item["flores_passage"]
    question = item["question"]
    mc_answer1 = item["mc_answer1"]
    mc_answer2 = item["mc_answer2"]
    mc_answer3 = item["mc_answer3"]
    mc_answer4 = item["mc_answer4"]

    formatted_question = (
        f"P: {flores_passage}\n"
        f"Q: {question.strip()}\n"
        f"A: {mc_answer1}\n"
        f"B: {mc_answer2}\n"
        f"C: {mc_answer3}\n"
        f"D: {mc_answer4}\n"
        f"Answer:"
    )
    return formatted_question

DATASET_CONFIGS = {
    "eusreading": {
        "loader": lambda: load_dataset("HiTZ/EusReading", split="test"),
        "formatter": format_eusreading_question,
    },
    "eusproficiency": {
        "loader": lambda: load_dataset("HiTZ/EusProficiency", name="default", split="test"),
        "formatter": format_eusproficiency_question,
    },
    "belebele": {
        "loader": lambda: load_dataset("facebook/belebele", name="eus_Latn", split="test"),
        "formatter": format_belebele_question,
    },
}

# ==================== HELPER FUNCTIONS ====================
def load_results(results_file):
    """Load results from jsonl file."""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def tokenize_text_static(text, tokenizer):
    """Tokenize text with static tokenizer."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": len(token_ids)
    }

def tokenize_text_dynamic(text, dynamic_bpe, max_merges=10):
    """Tokenize text with dynamic BPE."""
    # Import here to avoid circular imports
    from evaluation_helper_functions import dynamic_tokenize_texts
    
    # Get dynamic tokens
    dynamic_tokens = dynamic_tokenize_texts([text], dynamic_bpe, max_merges=max_merges)[0]
    
    return {
        "tokens": dynamic_tokens,
        "num_tokens": len(dynamic_tokens)
    }

def analyze_tokenization_difference(static_result, dynamic_result, text):
    """Analyze differences between static and dynamic tokenization."""
    analysis = {
        "text": text,
        "text_length": len(text),
        "static": static_result,
        "dynamic": dynamic_result,
        "token_count_diff": dynamic_result["num_tokens"] - static_result["num_tokens"],
        "compression_ratio_static": len(text) / static_result["num_tokens"] if static_result["num_tokens"] > 0 else 0,
        "compression_ratio_dynamic": len(text) / dynamic_result["num_tokens"] if dynamic_result["num_tokens"] > 0 else 0,
    }
    return analysis

def compare_results(results1, results2, dataset, formatter, tokenizer, dynamic_bpe):
    """Compare results between two models and analyze tokenization differences."""
    
    # Find disagreements
    disagreements = []
    agreements_correct = []
    agreements_incorrect = []
    
    for r1, r2 in zip(results1, results2):
        assert r1["id"] == r2["id"], f"ID mismatch: {r1['id']} vs {r2['id']}"
        
        if r1["prediction"] != r2["prediction"]:
            disagreements.append({
                "id": r1["id"],
                "gold": r1["gold"],
                "model1_pred": r1["prediction"],
                "model1_correct": r1["correct"],
                "model2_pred": r2["prediction"],
                "model2_correct": r2["correct"],
            })
        else:
            if r1["correct"]:
                agreements_correct.append(r1["id"])
            else:
                agreements_incorrect.append(r1["id"])
    
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples: {len(results1)}")
    print(f"Disagreements: {len(disagreements)} ({len(disagreements)/len(results1)*100:.2f}%)")
    print(f"Agreements (both correct): {len(agreements_correct)} ({len(agreements_correct)/len(results1)*100:.2f}%)")
    print(f"Agreements (both incorrect): {len(agreements_incorrect)} ({len(agreements_incorrect)/len(results1)*100:.2f}%)")
    
    # Analyze tokenization for disagreements
    detailed_analysis = []
    
    print(f"\n{'='*60}")
    print(f"ANALYZING TOKENIZATION FOR DISAGREEMENTS")
    print(f"{'='*60}\n")
    
    for idx, disagreement in enumerate(disagreements):
        item_id = disagreement["id"]
        item = dataset[item_id]
        
        # Format the question text
        question_text = formatter(item)
        
        # Tokenize with both methods
        static_tokens = tokenize_text_static(question_text, tokenizer)
        dynamic_tokens = tokenize_text_dynamic(question_text, dynamic_bpe, max_merges=10)
        
        # Analyze the difference
        tokenization_analysis = analyze_tokenization_difference(
            static_tokens, dynamic_tokens, question_text
        )
        
        # Combine with disagreement info
        detailed_item = {
            **disagreement,
            "question_text": question_text,
            "tokenization": tokenization_analysis
        }
        
        detailed_analysis.append(detailed_item)
        
        # Print summary for first few
        if idx < 5:
            print(f"Example {idx + 1} (ID: {item_id})")
            print(f"  Gold: {disagreement['gold']}")
            print(f"  Model 1 (Static): {disagreement['model1_pred']} {'✓' if disagreement['model1_correct'] else '✗'}")
            print(f"  Model 2 (Dynamic): {disagreement['model2_pred']} {'✓' if disagreement['model2_correct'] else '✗'}")
            print(f"  Tokens (Static): {static_tokens['num_tokens']}")
            print(f"  Tokens (Dynamic): {dynamic_tokens['num_tokens']}")
            print(f"  Difference: {dynamic_tokens['num_tokens'] - static_tokens['num_tokens']} tokens")
            print(f"  Compression ratio (Static): {tokenization_analysis['compression_ratio_static']:.2f}")
            print(f"  Compression ratio (Dynamic): {tokenization_analysis['compression_ratio_dynamic']:.2f}")
            print()
    
    return {
        "disagreements": disagreements,
        "detailed_analysis": detailed_analysis,
        "agreements_correct": agreements_correct,
        "agreements_incorrect": agreements_incorrect,
        "statistics": {
            "total": len(results1),
            "disagreements": len(disagreements),
            "agreements_correct": len(agreements_correct),
            "agreements_incorrect": len(agreements_incorrect),
        }
    }

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description="Compare model results and tokenization")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset to analyze")
    parser.add_argument("--results1", type=str, required=True,
                       help="Results file for model 1 (e.g., latxa_original)")
    parser.add_argument("--results2", type=str, required=True,
                       help="Results file for model 2 (e.g., latxa_dynamic)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file (default: comparison_<dataset>.json)")
    parser.add_argument("--tokenizer-path", type=str, 
                       default="HiTZ/latxa-7b-v1.2",
                       help="Path to static tokenizer")
    args = parser.parse_args()
    
    # Set default output file
    if args.output is None:
        args.output = f"comparison_{args.dataset}.json"
    
    print(f"Loading dataset: {args.dataset}")
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = dataset_config["loader"]()
    formatter = dataset_config["formatter"]
    
    print(f"Loading results from:")
    print(f"  Model 1: {args.results1}")
    print(f"  Model 2: {args.results2}")
    
    results1 = load_results(args.results1)
    results2 = load_results(args.results2)
    
    print(f"Results loaded: {len(results1)} examples")
    
    # Load tokenizers
    print("\nLoading static tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    print("Loading hypernet for dynamic tokenization...")
    hypernet_tokenizer = AutoTokenizer.from_pretrained(
        "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
    )
    
    dynamic_bpe = Dynamic_BPE(
        tokenizer=hypernet_tokenizer,
        tokenizer_boundary="pretokens",
    )
    
    # Compare results
    comparison = compare_results(
        results1, results2, dataset, formatter, tokenizer, dynamic_bpe
    )
    
    # Save to JSON
    print(f"\n{'='*60}")
    print(f"Saving detailed comparison to: {args.output}")
    print(f"{'='*60}\n")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Comparison saved!")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"TOKENIZATION STATISTICS (Disagreements Only)")
    print(f"{'='*60}")
    
    if comparison["detailed_analysis"]:
        token_diffs = [item["tokenization"]["token_count_diff"] 
                      for item in comparison["detailed_analysis"]]
        
        static_compression = [item["tokenization"]["compression_ratio_static"] 
                            for item in comparison["detailed_analysis"]]
        dynamic_compression = [item["tokenization"]["compression_ratio_dynamic"] 
                             for item in comparison["detailed_analysis"]]
        
        print(f"Average token count difference: {sum(token_diffs)/len(token_diffs):.2f}")
        print(f"Average static compression ratio: {sum(static_compression)/len(static_compression):.2f}")
        print(f"Average dynamic compression ratio: {sum(dynamic_compression)/len(dynamic_compression):.2f}")
        
        # Count cases where each model was correct
        model1_only_correct = sum(1 for item in comparison["detailed_analysis"] 
                                 if item["model1_correct"] and not item["model2_correct"])
        model2_only_correct = sum(1 for item in comparison["detailed_analysis"] 
                                 if item["model2_correct"] and not item["model1_correct"])
        both_incorrect = sum(1 for item in comparison["detailed_analysis"] 
                           if not item["model1_correct"] and not item["model2_correct"])
        
        print(f"\nDisagreement breakdown:")
        print(f"  Model 1 correct, Model 2 incorrect: {model1_only_correct}")
        print(f"  Model 2 correct, Model 1 incorrect: {model2_only_correct}")
        print(f"  Both incorrect: {both_incorrect}")

if __name__ == "__main__":
    main()