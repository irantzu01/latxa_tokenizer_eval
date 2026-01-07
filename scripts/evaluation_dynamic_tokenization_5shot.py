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
from datasets import load_dataset
import random
import torch
from tqdm import tqdm

ds  = load_dataset("HiTZ/EusProficiency", split="test")


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


def build_doc_text(item):
    return (
        f"Galdera: {item['question']}\n"
        f"A: {item['candidates'][0]}\n"
        f"B: {item['candidates'][1]}\n"
        f"C: {item['candidates'][2]}\n"
        f"D: {item['candidates'][3]}\n"
        f"Erantzuna:"
    )

def build_fewshot_context(ds, idx, k=5):
    ctx = []
    count = 0
    for j, ex in enumerate(ds):
        if j == idx:
            continue
        ctx.append(
            build_doc_text(ex) + " " + CHOICES[ex["answer"]]
        )
        count += 1
        if count == k:
            break
    return "\n\n".join(ctx)


import json
import torch
from tqdm import tqdm

CHOICES = ["A", "B", "C", "D"]

pad_id = latxa_tokenizer.pad_token_id
if pad_id is None:
    pad_id = latxa_tokenizer.eos_token_id


def dynamic_tokens_to_text(dynamic_tokens):
    text = ""
    for tok in dynamic_tokens:
        if tok.startswith("▁") or tok.startswith("Ġ"):
            text += " " + tok[1:]
        else:
            text += tok
    return text.strip()

# Tokenization and Caching
with open("cache/eusproficiency_dynamic_fewshot_tokenized.jsonl", "w") as f:
    for idx, item in enumerate(tqdm(ds, desc="Dynamic few-shot tokenization")):

        # ----- Build full prompt text -----
        fewshot_context = build_fewshot_context(ds, idx, k=5)
        prompt_text = fewshot_context + "\n\n" + build_doc_text(item)

        # ----- Dynamic tokenize prompt -----
        prompt_dyn = dynamic_tokenize_texts(
            [prompt_text],
            dynamic_bpe,
            batch_size=1
        )[0]

        prompt_text_recon = dynamic_tokens_to_text(prompt_dyn)

        prompt_ids = latxa_tokenizer.encode(
            prompt_text_recon,
            add_special_tokens=False
        )

        # ----- Dynamic tokenize each choice -----
        choice_ids = {}
        for c in CHOICES:
            dyn = dynamic_tokenize_texts(
                [" " + c],
                dynamic_bpe,
                batch_size=1
            )[0]

            recon = dynamic_tokens_to_text(dyn)

            ids = latxa_tokenizer.encode(
                recon,
                add_special_tokens=False
            )

            choice_ids[c] = ids

        f.write(json.dumps({
            "id": idx,
            "prompt_ids": prompt_ids,
            "choice_ids": choice_ids,
            "gold": item["answer"]
        }) + "\n")

print("Dynamic few-shot tokenization and caching completed.")
# # Evaluation
# correct = 0
# total = 0

# with open("cache/eusproficiency_dynamic_fewshot_tokenized.jsonl") as f:
#     for line in tqdm(f, desc="Evaluating (dynamic BPE)"):

#         item = json.loads(line)

#         full_ids = [
#             item["prompt_ids"] + item["choice_ids"][c]
#             for c in CHOICES
#         ]

#         # SAFETY CHECK (optional but recommended)
#         for seq in full_ids:
#             assert all(isinstance(x, int) for x in seq)

#         input_ids, attention_mask = build_batch_tensors(
#             full_ids, pad_id, device
#         )

#         scores = score_choices(model, input_ids, attention_mask)
#         pred = torch.argmax(scores).item()

#         if pred == item["gold"]:
#             correct += 1
#         total += 1

# accuracy = correct / total
# print(f"Dynamic BPE 5-shot accuracy: {accuracy:.4f}")

