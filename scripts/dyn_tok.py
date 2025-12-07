import sys
import os
import json


project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
morphscore_path = os.path.expanduser(
        "~/MASTER/WiSe25/Lab Rotation/morphscore"
)
sys.path.append(project_root)
sys.path.append(morphscore_path)


from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from zett.utils import get_surface_form_matrix
from datasets import load_dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import OrderedDict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Latxa model and tokenizer
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
print("Latxa model and tokenizer loaded.")


# Load hypernetwork
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental",
    trust_remote_code=True
)
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)

dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)
print("Hypernetwork + tokenizer + Dynamic BPE ready.")


# Load datasets
ds_EusProficiency = load_dataset("HiTZ/EusProficiency")
#ds_EusTrivia = load_dataset("HiTZ/EusTrivia")
#ds_EusReading = load_dataset("HiTZ/EusReading")
#ds_EusExams = load_dataset("HiTZ/EusExams")
questions = ds_EusProficiency["test"]["question"]
candidates_list = ds_EusProficiency["test"]["candidates"]
print("Dataset loaded.")


# Build raw examples for dynamic tokenization
raw_examples = []
for q, cand_list in zip(questions, candidates_list):
    combined = q + " [QSEP] " + " [CSEP] ".join(cand_list)
    raw_examples.append({"text": combined})   
print(raw_examples[0])


# Tokenize raw examples with Latxa tokenizer
encoded_latxa = []
for example in raw_examples:
    enc = latxa_tokenizer(
        example["text"],
        truncation=True,
        padding=False,
        return_tensors=None
    )
    encoded_latxa.append(enc)
print("Latxa tokenization completed.")
print(encoded_latxa[0])


# Dynamic BPE tokenization
encoded_dynamic = []
BATCH_SIZE = 500
for i in tqdm(range(0, len(raw_examples), BATCH_SIZE), desc="Dynamic BPE"):
    batch = raw_examples[i : i + BATCH_SIZE]
    dynamic_tokens, attr2, attr3, attr4 = dynamic_bpe.tokenize_batch(
        batch_examples=batch,
        max_nr_merges=10,
        mlm=True
    )
    encoded_dynamic.extend(dynamic_tokens)
print("Dynamic BPE tokenization completed.")
print("Number of dynamic tokenized examples:", len(encoded_dynamic))
print(encoded_dynamic[0])


# Evaluation
from simple_tokenization_evaluation import (
    avg_tokens_per_word,
    token_length_distribution,
    plot_token_length_distribution,
    token_set_overlap,
    word_coverage_latxa,
    word_coverage_dynamic
)

encoded_latxa_tokens = [latxa_tokenizer.convert_ids_to_tokens(enc["input_ids"]) for enc in encoded_latxa]

# avg_latxa = avg_tokens_per_word(raw_examples, encoded_latxa_tokens)
# avg_dyn = avg_tokens_per_word(raw_examples, encoded_dynamic)
# print("Average tokens per word and per sentence:")
# print("Latxa:", avg_latxa)
# print("Dynamic BPE:", avg_dyn)

coverage_latxa = word_coverage_latxa(raw_examples, encoded_latxa)
coverage_dyn = word_coverage_dynamic(raw_examples, encoded_dynamic)
print("Word coverage in latxa:", coverage_latxa)
print("Word coverage in dynamic BPE:", coverage_dyn)


# overlap = token_set_overlap(encoded_latxa, encoded_dynamic)


    

