import sys
import os
import json



project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)


from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from zett.utils import get_surface_form_matrix
from collections import Counter
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import OrderedDict



def format_prompt(question, candidates):
    return (
        f"Galdera: {question}\n"
        f"A: {candidates[0]}\n"
        f"B: {candidates[1]}\n"
        f"C: {candidates[2]}\n"
        f"D: {candidates[3]}\n"
        f"Erantzuna:"
    )


#Load the EusProficiency dataset and prepare evaluation items
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
print(evaluation_items[0])
print("Evaluation items prepared:", len(evaluation_items))



def dynamic_tokenize_texts(texts, dynamic_bpe, batch_size=128, max_merges=10):
    """
    texts: list[str]
    returns: list[list[str]]  (dynamic tokens per text)
    """
    all_tokens = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Dynamic BPE"):
        batch_texts = texts[i:i+batch_size]
        batch_examples = [{"text": t} for t in batch_texts]

        dyn_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
            batch_examples=batch_examples,
            max_nr_merges=max_merges,
            mlm=True
        )

        all_tokens.extend(dyn_tokens)

    return all_tokens


for item in evaluation_items[:1]:  # start with 1 item for sanity check
    dynamic_choice_tokens = dynamic_tokenize_texts(
        item["choice_texts"],
        dynamic_bpe,
        batch_size=4
    )

    item["dynamic_tokens"] = dynamic_choice_tokens

print(item["dynamic_tokens"][0][:30])

len(evaluation_items)               # == dataset size
len(evaluation_items[0]["choice_texts"])  # == 4
len(evaluation_items[0]["dynamic_tokens"]) # == 4