import sys
import os
import json

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from evaluation_helper_functions import (
    dynamic_tokenize_texts,
    dynamic_tokens_to_latxa_ids,
    build_batch_tensors,
    score_choices
)


# Load model and tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
model.to(device)
model.eval()
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)
dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)


def format_prompt_reading(context, question, candidates):
    """
    Builds a multiple-choice prompt with a variable number of options.
    """
    letters = ["A", "B", "C", "D", "E", "F"]

    prompt = f"Pasartea: {context}\n\n"
    prompt += f"Galdera: {question}\n"

    for i, opt in enumerate(candidates):
        if opt is None or opt.strip() == "":
            continue
        prompt += f"{letters[i]}. {opt}\n"

    prompt += "Erantzuna:"
    return prompt


# Load EusReading dataset
ds = load_dataset("HiTZ/EusReading", split="test")

evaluation_items = []

for item in ds:
    candidates = item["candidates"]  # <-- confirm key name, usually "options"
    # Remove empty options
    candidates = [o for o in candidates if o and o.strip()]
    if len(candidates) < 2:
        continue  # skip broken items
    prompt = format_prompt_reading(
        context=item["context"],
        question=item["question"],
        candidates=candidates
    )
    choice_texts = []
    letters = [" A", " B", " C", " D", " E", " F"]
    for i in range(len(candidates)):
        choice_texts.append(prompt + letters[i])
    evaluation_items.append({
        "choice_texts": choice_texts,
        "answer": item["answer"]  # integer index
    })
print("Prepared EusReading items:", len(evaluation_items))


for item in tqdm(evaluation_items, desc="Dynamic BPE"):
    dynamic_tokens = dynamic_tokenize_texts(
        item["choice_texts"],
        dynamic_bpe,
        batch_size=4
    )
    item["dynamic_tokens"] = dynamic_tokens
print("Dynamic tokenization completed.")


# Evaluation loop
pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id
correct = 0
for item in tqdm(evaluation_items, desc="Evaluating"):
    choice_ids = dynamic_tokens_to_latxa_ids(
        item["dynamic_tokens"],
        latxa_tokenizer
    )
    input_ids, attention_mask = build_batch_tensors(
        choice_ids,
        pad_id,
        device
    )
    scores = score_choices(model, input_ids, attention_mask)
    pred = torch.argmax(scores).item()
    if pred == item["answer"]:
        correct += 1

accuracy = correct / len(evaluation_items)
print(f"\nFinal accuracy on EusReading (Dynamic BPE segmentation + Latxa vocab): {accuracy:.4f}")
