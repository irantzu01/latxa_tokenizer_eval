import sys
import os

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)


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
print("Datasets loaded.")
questions = ds_EusProficiency["test"]["question"]
candidates_list = ds_EusProficiency["test"]["candidates"]


encoded_questions = []
encoded_candidates = []

for question, cand_list in zip(questions, candidates_list):
    tokenized_q = []
    for cand in cand_list:
        enc = latxa_tokenizer(
            question,
            cand,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        tokenized_q.append(enc)

    encoded_candidates.append(tokenized_q)
print("Questions and candidates tokenized with Latxa tokenizer.")
print("Number of questions:", len(encoded_candidates))
print(encoded_candidates[0:2])

# Get dynamic tokenization




    

