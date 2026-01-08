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

def build_fewshot_example(item):
    answer_letter = ["A", "B", "C", "D"][item["answer"]]
    return build_doc_text(item) + " " + answer_letter

def build_fewshot_context(ds, current_idx, k=5, seed=42):
    rng = random.Random(seed + current_idx)
    # pool excludes current example
    pool = [ds[i] for i in range(len(ds)) if i != current_idx]
    fewshot_examples = rng.sample(pool, k)
    texts = [build_fewshot_example(ex) for ex in fewshot_examples]
    return "\n\n".join(texts)



CHOICES = [" A", " B", " C", " D"]

pad_id = latxa_tokenizer.pad_token_id
if pad_id is None:
    pad_id = latxa_tokenizer.eos_token_id

with open("cache/eusproficiency_latxa_fewshot_tokenized.jsonl", "w") as f, \
     open("cache/eusproficiency_latxa_queries.jsonl", "w") as f2:
    for idx, item in enumerate(tqdm(ds, desc="Tokenizing (Latxa few-shot)")):
        fewshot_context = build_fewshot_context(ds, idx, k=5)
        query_text = build_doc_text(item)
        prompt_text = fewshot_context + "\n\n" + query_text
        prompt_tokens = latxa_tokenizer.tokenize(prompt_text)
        query_tokens = latxa_tokenizer.tokenize(query_text)       
        choice_tokens = {
            c: latxa_tokenizer.tokenize(" " + c)
            for c in ["A", "B", "C", "D"]
        }
        f.write(json.dumps({
            "id": idx,
            "prompt_tokens": prompt_tokens,
            "choice_tokens": choice_tokens,
            "gold": item["answer"]
        }) + "\n")
        f2.write(json.dumps({
            "id": idx,
            "query_tokens": query_tokens,
            "gold": item["answer"]
        }) + "\n")

print("Latxa few-shot tokenization (tokens) cached.")


# correct = 0
# total = 0

# results_path = "cache/eusproficiency_latxa_eval_results.jsonl"

# with open("cache/eusproficiency_latxa_fewshot_tokenized.jsonl") as fin, \
#      open(results_path, "w") as fout:

#     for line in tqdm(fin, desc="Evaluating (Latxa few-shot)"):

#         item = json.loads(line)

#         # ----- Reconstruct prompt -----
#         prompt_text = latxa_tokenizer.convert_tokens_to_string(
#             item["prompt_tokens"]
#         )

#         prompt_ids = latxa_tokenizer.encode(
#             prompt_text,
#             add_special_tokens=False
#         )

#         # ----- Build full sequences -----
#         full_ids = []
#         for c in ["A", "B", "C", "D"]:
#             choice_text = latxa_tokenizer.convert_tokens_to_string(
#                 item["choice_tokens"][c]
#             )

#             choice_ids = latxa_tokenizer.encode(
#                 choice_text,
#                 add_special_tokens=False
#             )

#             full_ids.append(prompt_ids + choice_ids)

#         # ----- Batch + score -----
#         input_ids, attention_mask = build_batch_tensors(
#             full_ids, pad_id, device
#         )

#         scores = score_choices(model, input_ids, attention_mask)
#         pred = torch.argmax(scores).item()
#         is_correct = (pred == item["gold"])

#         # ----- Accumulate -----
#         if is_correct:
#             correct += 1
#         total += 1

#         # ----- Save instance result -----
#         fout.write(json.dumps({
#             "id": item["id"],
#             "gold": item["gold"],
#             "prediction": pred,
#             "correct": is_correct,
#             "scores": scores.tolist()
#         }) + "\n")

# accuracy = correct / total
# print(f"Latxa 5-shot (LM Eval style) accuracy: {accuracy:.4f}")
# print(f"Saved per-instance results to {results_path}")
