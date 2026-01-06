import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from evaluation_helper_functions import (
    dynamic_tokenize_texts,
    dynamic_tokens_to_latxa_ids,
    build_batch_tensors,
    score_choices
)


from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# Load model and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
model.to(device)
model.eval()
print("Latxa model and tokenizer loaded.")

hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)
dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)


#Load the EusProficiency dataset and prepare evaluation items
def format_prompt(question, candidates):
    return (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )

ds = load_dataset("HiTZ/EusProficiency", split="test")

CHOICES = [" A", " B", " C", " D"]
evaluation_items = []
for item in ds:
    prompt = format_prompt(item["question"], item["candidates"])
    choice_texts = [prompt + choice for choice in CHOICES]
    evaluation_items.append({
        "prompt": prompt,
        "choice_texts": choice_texts,   # length 4
        "answer": item["answer"]        # int: 0–3
    })

for item in tqdm(evaluation_items, desc="Dynamic BPE"):
    dynamic_tokens = dynamic_tokenize_texts(
        item["choice_texts"],
        dynamic_bpe,
        batch_size=4
    )
    item["dynamic_tokens"] = dynamic_tokens
    latxa_tokens = [
        latxa_tokenizer.tokenize(text)
        for text in item["choice_texts"]
    ]
    item["latxa_tokens"] = latxa_tokens
print("Dynamic tokenization and latxa tokenization completed.")

for i, tokens in enumerate(latxa_tokens):
    print(item["choice_texts"][i])
    print(tokens)

# # Evaluation loop for Latxa with Dynamic BPE segmentation
# tokenized_dyn_out = open("cache/EusProficiency_dynamic_tokenized.jsonl", "w")
# pred_dyn_out = open("results/EusProficiency_dynamic_predictions.jsonl", "w")
# pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id
# correct = 0
# for idx, item in enumerate(tqdm(evaluation_items, desc="Evaluating")):
#     choice_ids = dynamic_tokens_to_latxa_ids(
#         item["dynamic_tokens"],
#         latxa_tokenizer
#     )
#     # save tokenization ONCE
#     tokenized_dyn_out.write(json.dumps({
#         "id": idx,
#         "dynamic_tokens": item["dynamic_tokens"],
#         "choice_ids": choice_ids,          # list[list[int]]
#         "answer": item["answer"]
#     }) + "\n")
#     input_ids, attention_mask = build_batch_tensors(
#         choice_ids,
#         pad_id,
#         device
#     )
#     scores = score_choices(model, input_ids, attention_mask)
#     pred = torch.argmax(scores).item()
#     pred_dyn_out.write(json.dumps({
#     "id": idx,
#     "scores": scores.tolist(),
#     "prediction": pred,
#     "gold": item["answer"],
#     "correct": pred == item["answer"]
#     }) + "\n")
#     if pred == item["answer"]:
#         correct += 1

# accuracy_bpe = correct / len(evaluation_items)
# print(f"\nFinal accuracy on EusProficiency (Dynamic BPE segmentation + Latxa vocab): {accuracy_bpe:.4f}")
# tokenized_dyn_out.close()
# pred_dyn_out.close()

# Evaluation loop for Latxa with its own tokenization
tokenized_latxa_out = open("cache/EusProficiency_latxa_tokenized_prueba.jsonl", "w")
pred_latxa_out = open("results/EusProficiency_latxa_predictions_prueba.jsonl", "w")
pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id
correct = 0

for idx, item in enumerate(tqdm(evaluation_items, desc="Evaluating Latxa")):
    choice_ids = [
        latxa_tokenizer.encode("".join(tokens), add_special_tokens=False)
        for tokens in item["latxa_tokens"]
    ]
    # save tokenization
    tokenized_latxa_out.write(json.dumps({
        "id": idx,
        "latxa_tokens": item["latxa_tokens"],
        "choice_ids": choice_ids,
        "answer": item["answer"]
    }) + "\n")
    # optional: score only the answer portion
    if "prompt_len" in item:
        answer_choice_ids = [ids[item["prompt_len"]:] for ids in choice_ids]
    else:
        answer_choice_ids = choice_ids
    input_ids, attention_mask = build_batch_tensors(
        answer_choice_ids,
        pad_id,
        device
    )
    scores = score_choices(model, input_ids, attention_mask)
    # length normalization
    scores = scores / attention_mask.sum(dim=1)
    pred = torch.argmax(scores).item()
    pred_latxa_out.write(json.dumps({
        "id": idx,
        "scores": scores.tolist(),
        "prediction": pred,
        "gold": item["answer"],
        "correct": pred == item["answer"]
    }) + "\n")
    if pred == item["answer"]:
        correct += 1

accuracy_latxa = correct / len(evaluation_items)
print(f"\nFinal accuracy on EusProficiency (Latxa own tokenization): {accuracy_latxa:.4f}")

tokenized_latxa_out.close()
pred_latxa_out.close()
