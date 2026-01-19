#!/usr/bin/env python3
"""
Analyze answer distribution (A, B, C, D) across multiple model results files.
Shows if models have biases toward certain answer choices.
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def load_results(filepath):
    """Load results from a JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_predictions(results):
    """Analyze prediction distribution."""
    predictions = [r['prediction'] for r in results]
    golds = [r['gold'] for r in results]
    
    pred_counts = Counter(predictions)
    gold_counts = Counter(golds)
    
    total = len(predictions)
    
    # Calculate percentages
    pred_percentages = {choice: (count / total) * 100 
                       for choice, count in pred_counts.items()}
    gold_percentages = {choice: (count / total) * 100 
                       for choice, count in gold_counts.items()}
    
    # Calculate accuracy per choice
    accuracy_per_choice = {}
    for choice in set(predictions + golds):
        correct = sum(1 for r in results if r['prediction'] == choice and r['correct'])
        total_pred = pred_counts.get(choice, 0)
        if total_pred > 0:
            accuracy_per_choice[choice] = (correct / total_pred) * 100
        else:
            accuracy_per_choice[choice] = 0.0
    
    return {
        'pred_counts': pred_counts,
        'gold_counts': gold_counts,
        'pred_percentages': pred_percentages,
        'gold_percentages': gold_percentages,
        'accuracy_per_choice': accuracy_per_choice,
        'total': total
    }


def format_model_name(filepath):
    """Extract a readable model name from filepath."""
    filename = Path(filepath).stem
    # Remove dataset prefix and shot suffix
    parts = filename.split('_')
    # Find where model name starts (after dataset name)
    if 'latxa' in filename:
        idx = parts.index('latxa') if 'latxa' in parts else 1
        model_parts = parts[idx:]
        # Remove shot info
        model_parts = [p for p in model_parts if 'shot' not in p]
        return '_'.join(model_parts)
    return filename


def print_summary_table(analyses):
    """Print a summary table of all models."""
    
    # Get all unique choices across all models
    all_choices = set()
    for analysis in analyses.values():
        all_choices.update(analysis['pred_percentages'].keys())
        all_choices.update(analysis['gold_percentages'].keys())
    
    all_choices = sorted(all_choices)
    choice_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
    
    print("\n" + "="*100)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*100)
    
    # Header
    print(f"\n{'Model':<35}", end="")
    print(f"{'Total':<8}", end="")
    for choice in all_choices:
        label = choice_labels.get(choice, str(choice))
        print(f"{label:>10}", end="")
    print()
    print("-" * 100)
    
    # Gold distribution (only once)
    if analyses:
        first_analysis = list(analyses.values())[0]
        print(f"{'GOLD DISTRIBUTION':<35}", end="")
        print(f"{first_analysis['total']:<8}", end="")
        for choice in all_choices:
            pct = first_analysis['gold_percentages'].get(choice, 0.0)
            print(f"{pct:>9.1f}%", end="")
        print("\n")
    
    # Each model's predictions
    for model_name, analysis in analyses.items():
        print(f"{model_name:<35}", end="")
        print(f"{analysis['total']:<8}", end="")
        for choice in all_choices:
            pct = analysis['pred_percentages'].get(choice, 0.0)
            print(f"{pct:>9.1f}%", end="")
        print()
    
    print("\n" + "="*100)
    print("ACCURACY PER ANSWER CHOICE")
    print("="*100)
    print(f"\n{'Model':<35}", end="")
    for choice in all_choices:
        label = choice_labels.get(choice, str(choice))
        print(f"{label:>10}", end="")
    print()
    print("-" * 100)
    
    for model_name, analysis in analyses.items():
        print(f"{model_name:<35}", end="")
        for choice in all_choices:
            acc = analysis['accuracy_per_choice'].get(choice, 0.0)
            print(f"{acc:>9.1f}%", end="")
        print()
    
    # Bias analysis
    print("\n" + "="*100)
    print("BIAS ANALYSIS (Prediction % - Gold %)")
    print("="*100)
    print(f"\n{'Model':<35}", end="")
    for choice in all_choices:
        label = choice_labels.get(choice, str(choice))
        print(f"{label:>10}", end="")
    print()
    print("-" * 100)
    
    gold_dist = list(analyses.values())[0]['gold_percentages']
    for model_name, analysis in analyses.items():
        print(f"{model_name:<35}", end="")
        for choice in all_choices:
            pred_pct = analysis['pred_percentages'].get(choice, 0.0)
            gold_pct = gold_dist.get(choice, 0.0)
            bias = pred_pct - gold_pct
            print(f"{bias:>+9.1f}%", end="")
        print()
    
    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze answer distribution across multiple model result files"
    )
    parser.add_argument(
        "files",
        nargs='+',
        type=str,
        help="One or more result JSONL files to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save detailed statistics to JSON file"
    )
    
    args = parser.parse_args()
    
    analyses = {}
    
    # Analyze each file
    for filepath in args.files:
        print(f"Loading {filepath}...")
        results = load_results(filepath)
        model_name = format_model_name(filepath)
        analyses[model_name] = analyze_predictions(results)
    
    # Print summary table
    print_summary_table(analyses)
    
    # Save detailed stats if requested
    if args.output:
        detailed_stats = {}
        for model_name, analysis in analyses.items():
            detailed_stats[model_name] = {
                'total_predictions': analysis['total'],
                'prediction_counts': dict(analysis['pred_counts']),
                'prediction_percentages': analysis['pred_percentages'],
                'gold_counts': dict(analysis['gold_counts']),
                'gold_percentages': analysis['gold_percentages'],
                'accuracy_per_choice': analysis['accuracy_per_choice']
            }
        
        with open(args.output, 'w') as f:
            json.dump(detailed_stats, f, indent=2)
        print(f"✓ Detailed statistics saved to: {args.output}")


if __name__ == "__main__":
    main()