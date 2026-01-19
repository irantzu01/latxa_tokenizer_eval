#!/usr/bin/env python3
"""
Evaluate a single model on all BasqueGLUE tasks.
Choose model via command line argument.
"""

import sys
import os
import json
import random
import html
import re
import argparse
from datasets import load_dataset, load_metric
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

SHOTS = 5
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# ==================== HELPER FUNCTIONS ====================
def general_detokenize(string):
    string = re.sub(r'\s+([.,;:!?)])', r'\1', string)
    string = re.sub(r'(\s+|^)\(\s+([^)]+)\s+\)', r'\1(\2)', string)
    string = re.sub(r'(\s+|^)\[\s+([^)]+)\s+\]', r'\1[\2]', string)
    string = re.sub(r'(\s+|^)"\s+([^"]+)\s+"', r'\1"\2"', string)
    string = re.sub(r"(\s+|^)'\s+([^']+)\s+'", r"\1'\2'", string)
    return string

def process_doc(string):
    string = html.unescape(string)
    string = general_detokenize(string)
    return string

def process_wic_docs(dataset):
    def _helper(doc):
        doc["sentence1"] = process_doc(doc["sentence1"]).encode('latin-1').decode('utf-8')
        doc["sentence2"] = process_doc(doc["sentence2"]).encode('latin-1').decode('utf-8')
        return doc
    return dataset.map(_helper)

def format_question_bec(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak?\nErantzuna:"

def format_question_bhtc(item):
    labels = ', '.join(CONFIGS['bhtc'][2])
    return f"Testua: {item['text']}\nGaldera: Zein da aurreko testuaren gaia? Aukeratu hauen artean: {labels}\nErantzuna:"

def format_question_coref(item):
    def _span_in_context(span_index, span_text):
        span_start = span_index
        span_end = span_start + len(span_text.split(" ")) - 1
        tokens[span_start] = f'*{tokens[span_start]}'
        tokens[span_end] = f'{tokens[span_end]}*'
    tokens = item["text"].split(" ")
    _span_in_context(item["span1_index"], item["span1_text"])
    _span_in_context(item["span2_index"] - 1, item["span2_text"])
    context = process_doc(" ".join(tokens))
    span_1 = process_doc(item["span1_text"])
    span_2 = process_doc(item["span2_text"])
    text = (
        f'Testua: {context}\nGaldera: Aurreko testuan, "*{span_1}*" eta "*{span_2}*" gauza bera dira?\nErantzuna:'
    )
    return text

def format_question_qnli(item):
    return f"{item['question']}\n{item['sentence']}\nGaldera: aurreko galderari erantzuten al dio emandako testuak?\nErantzuna:"

def format_question_vaxx(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak txertoei buruz?\nErantzuna:"

def format_question_wic(item):
    return f"1. esaldia: {item['sentence1']}\n2. esaldia: {item['sentence2']}\nGaldera: Aurreko bi esaldietan, \"{item['word']}\" hitzak esanahi berdina du?\nErantzuna:"

def build_fewshot_example(item, format_question_func, labels):
    """Build a complete example with answer."""
    question = format_question_func(item)
    answer = labels[item['label']] if isinstance(item['label'], int) else item['label']
    return question + " " + answer

def build_fewshot_context(dataset_list, current_idx, format_question_func, labels, k=5):
    """Build k-shot context excluding current item."""
    pool = [dataset_list[i] for i in range(len(dataset_list)) if i != current_idx]
    few_shot_examples = random.sample(pool, min(k, len(pool)))
    texts = [build_fewshot_example(ex, format_question_func, labels) for ex in few_shot_examples]
    return "\n\n".join(texts)

def micro_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="micro")["f1"]
    return f1_score

def vaxx_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, labels=[0, 2], average="macro")["f1"]
    return f1_score

def accuracy_score(items):
    arr = [int(g == p) for g, p in items]
    return sum(arr) / len(arr)

# ==================== TASK CONFIGURATIONS ====================
CONFIGS = {
    "bec": (format_question_bec, micro_f1_score, ['negatiboa', 'neutrala', 'positiboa']),
    "bhtc": (format_question_bhtc, micro_f1_score, [
        'Ekonomia', 'Euskal Herria', 'Euskara', 'Gizartea', 'Historia', 'Ingurumena', 
        'Iritzia', 'Komunikazioa', 'Kultura', 'Nazioartea', 'Politika', 'Zientzia'
    ]),
    "coref": (format_question_coref, accuracy_score, ['ez', 'bai']),
    "qnli": (format_question_qnli, accuracy_score, ['bai', 'ez']),
    "vaxx": (format_question_vaxx, vaxx_f1_score, ['aurka', 'neutrala', 'alde']),
    "wic": (format_question_wic, accuracy_score, ['ez', 'bai'])
}

# ==================== EVALUATION FUNCTIONS ====================
def evaluate_static_model(model, tokenizer, dataset, task, device='cuda'):
    """Evaluate a model with static tokenization."""
    format_question_func, eval_func, possible_answers = CONFIGS[task]
    
    y_gold_and_pred = []
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    # Convert to list for proper indexing
    dataset_list = list(dataset)
    
    for idx, item in enumerate(tqdm(dataset_list, desc=f"  {task}")):
        fewshot_context = build_fewshot_context(
            dataset_list, idx, format_question_func, possible_answers, k=SHOTS
        )
        query_text = format_question_func(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        
        full_ids = []
        for answer in possible_answers:
            answer_text = " " + answer
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
            full_ids.append(prompt_ids + answer_ids)
        
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        
        # CRITICAL: Use torch.no_grad() for inference
        with torch.no_grad():
            scores = score_choices(model, input_ids, attention_mask)
        
        pred_idx = torch.argmax(scores).item()
        gold_idx = item["label"]
        
        y_gold_and_pred.append((gold_idx, pred_idx))
        
        results.append({
            "id": idx,
            "gold": gold_idx,
            "prediction": pred_idx,
            "correct": (pred_idx == gold_idx),
            "scores": scores.tolist()
        })
    
    score = eval_func(y_gold_and_pred)
    return score, results

def evaluate_dynamic_model(model, tokenizer, augmenter, dynamic_bpe, dataset, task, device='cuda'):
    """Evaluate a model with dynamic tokenization."""
    format_question_func, eval_func, possible_answers = CONFIGS[task]
    
    y_gold_and_pred = []
    results = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    
    from evaluation_helper_functions import dynamic_tokenize_texts
    
    # Convert to list for proper indexing
    dataset_list = list(dataset)
    
    for idx, item in enumerate(tqdm(dataset_list, desc=f"  {task} (dynamic)")):
        fewshot_context = build_fewshot_context(
            dataset_list, idx, format_question_func, possible_answers, k=SHOTS
        )
        query_text = format_question_func(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        
        prompt_tokens = dynamic_tokenize_texts([prompt_text], dynamic_bpe, max_merges=10)[0]
        prompt_ids = augmenter.tokens_to_ids([prompt_tokens])[0]
        
        full_ids = []
        for answer in possible_answers:
            answer_text = " " + answer
            answer_tokens = dynamic_tokenize_texts([answer_text], dynamic_bpe, max_merges=10)[0]
            answer_ids = augmenter.tokens_to_ids([answer_tokens])[0]
            full_ids.append(prompt_ids + answer_ids)
        
        input_ids, attention_mask = build_batch_tensors(full_ids, pad_id, device)
        
        # CRITICAL: Use torch.no_grad() for inference
        with torch.no_grad():
            scores = score_choices(model, input_ids, attention_mask)
        
        pred_idx = torch.argmax(scores).item()
        gold_idx = item["label"]
        
        y_gold_and_pred.append((gold_idx, pred_idx))
        
        results.append({
            "id": idx,
            "gold": gold_idx,
            "prediction": pred_idx,
            "correct": (pred_idx == gold_idx),
            "scores": scores.tolist()
        })
    
    score = eval_func(y_gold_and_pred)
    return score, results

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODELS.keys()),
                       help="Model to evaluate")
    parser.add_argument("--task", type=str, default=None,
                       choices=list(CONFIGS.keys()),
                       help="Specific task to evaluate (default: all tasks)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = args.model
    model_config = MODELS[model_name]
    
    print(f"\n{'='*80}")
    print(f"LOADING MODEL: {model_config['description']}")
    print(f"{'='*80}")
    
    # Load model once
    model = AutoModelForCausalLM.from_pretrained(model_config["path"])
    tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_path"])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Initialize dynamic augmenter if needed
    augmenter = None
    dynamic_bpe = None
    if model_config["use_dynamic"]:
        print("Initializing dynamic components...")
        hypernet = AutoModel.from_pretrained(
            "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental",
            trust_remote_code=True
        )
        hypernet_tokenizer = AutoTokenizer.from_pretrained(
            "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
        )
        dynamic_bpe = Dynamic_BPE(
            tokenizer=hypernet_tokenizer,
            tokenizer_boundary="pretokens"
        )
        augmenter = DynamicAugmenter(
            model=model,
            latxa_tokenizer=tokenizer,
            hypernet=hypernet,
            hypernet_tokenizer=hypernet_tokenizer,
            cache_limit=50000,
            device=device
        )
        print("✓ Dynamic components loaded")
    
    # Determine which tasks to evaluate
    tasks_to_eval = [args.task] if args.task else list(CONFIGS.keys())
    
    model_results = {}
    os.makedirs("results", exist_ok=True)
    
    # Evaluate on tasks
    for task_name in tasks_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating task: {task_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = load_dataset("orai-nlp/basqueGLUE", name=task_name, split="test")
        if task_name == 'wic':
            dataset = process_wic_docs(dataset)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Evaluate
        if model_config["use_dynamic"]:
            score, results = evaluate_dynamic_model(
                model, tokenizer, augmenter, dynamic_bpe, dataset, task_name, device=device
            )
        else:
            score, results = evaluate_static_model(
                model, tokenizer, dataset, task_name, device=device
            )
        
        model_results[task_name] = score
        
        # Save detailed results
        results_file = f"results/basqueglue_{task_name}_{model_name}_{SHOTS}shot.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        
        _, eval_func, _ = CONFIGS[task_name]
        metric_name = eval_func.__name__.replace('_', ' ').title()
        print(f"\n✓ {task_name} - {metric_name}: {score:.4f}")
        print(f"  Results saved to: {results_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY: {model_config['description']}")
    print(f"{'='*80}")
    
    for task_name, score in model_results.items():
        _, eval_func, _ = CONFIGS[task_name]
        metric_name = eval_func.__name__.replace('_', ' ').title()
        print(f"{task_name:>10}: {score:.4f} ({metric_name})")
    
    # Save summary
    summary_file = f"results/basqueglue_{model_name}_{SHOTS}shot_summary.json"
    summary = {
        "model": model_name,
        "description": model_config['description'],
        "shots": SHOTS,
        "results": model_results
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()