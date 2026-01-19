#!/usr/bin/env python3
"""
Clean comparison JSON file by:
1. Remove the "disagreements" summary list
2. Keep only "detailed_analysis"
3. Remove "question_text" from each item
4. Remove "token_ids" from tokenization
5. Format tokens as single-line strings instead of lists
"""

import json
import argparse


def clean_comparison_data(data):
    """Clean the comparison data structure."""
    
    # Start with only detailed_analysis
    cleaned = {
        "detailed_analysis": []
    }
    
    # Copy statistics if present
    if "statistics" in data:
        cleaned["statistics"] = data["statistics"]
    
    # Process each item in detailed_analysis
    for item in data.get("detailed_analysis", []):
        cleaned_item = {
            "id": item["id"],
            "gold": item["gold"],
            "model1_pred": item["model1_pred"],
            "model1_correct": item["model1_correct"],
            "model2_pred": item["model2_pred"],
            "model2_correct": item["model2_correct"],
        }
        
        # Process tokenization if present
        if "tokenization" in item:
            tokenization = item["tokenization"]
            
            cleaned_tokenization = {
                "text": tokenization.get("text", ""),
                "text_length": tokenization.get("text_length", 0),
            }
            
            # Process static tokens
            if "static" in tokenization:
                static = tokenization["static"]
                # Join tokens into a single string
                tokens_str = " ".join(static.get("tokens", []))
                cleaned_tokenization["static"] = {
                    "tokens": tokens_str,
                    "num_tokens": static.get("num_tokens", 0)
                }
            
            # Process dynamic tokens
            if "dynamic" in tokenization:
                dynamic = tokenization["dynamic"]
                # Join tokens into a single string
                tokens_str = " ".join(dynamic.get("tokens", []))
                cleaned_tokenization["dynamic"] = {
                    "tokens": tokens_str,
                    "num_tokens": dynamic.get("num_tokens", 0)
                }
            
            # Add compression ratios and diff
            if "token_count_diff" in tokenization:
                cleaned_tokenization["token_count_diff"] = tokenization["token_count_diff"]
            if "compression_ratio_static" in tokenization:
                cleaned_tokenization["compression_ratio_static"] = tokenization["compression_ratio_static"]
            if "compression_ratio_dynamic" in tokenization:
                cleaned_tokenization["compression_ratio_dynamic"] = tokenization["compression_ratio_dynamic"]
            
            cleaned_item["tokenization"] = cleaned_tokenization
        
        cleaned["detailed_analysis"].append(cleaned_item)
    
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Clean comparison JSON file by removing unnecessary fields and compacting tokens"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <input>_cleaned.json)"
    )
    
    args = parser.parse_args()
    
    # Set output filename
    if args.output is None:
        if args.input_file.endswith('.json'):
            args.output = args.input_file.replace('.json', '_cleaned.json')
        else:
            args.output = args.input_file + '_cleaned.json'
    
    print(f"Reading from: {args.input_file}")
    
    # Read input JSON
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Original data has {len(data.get('disagreements', []))} disagreements")
    print(f"Original data has {len(data.get('detailed_analysis', []))} detailed items")
    
    # Clean the data
    cleaned_data = clean_comparison_data(data)
    
    print(f"Cleaned data has {len(cleaned_data['detailed_analysis'])} detailed items")
    
    # Write output JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Cleaned data saved to: {args.output}")
    
    # Show example of first item
    if cleaned_data['detailed_analysis']:
        print("\nExample of first cleaned item:")
        print(json.dumps(cleaned_data['detailed_analysis'][0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()