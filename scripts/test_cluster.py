from transformers import AutoTokenizer
import re
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

model_name = "HiTZ/latxa-7b-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ds_euscrawl = load_dataset("HiTZ/latxa-corpus-v1.1", 'euscrawl-v1.1')

text = "Euskara adimen arttifizialera iritsi da!"

# Encode text → token IDs
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded text:", decoded_text)


def avg_tokens_per_word(sentences):
    token_counts = []
    word_counts = []

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)
        words = sent.split()

        token_counts.append(len(tokens))
        word_counts.append(len(words))

    avg_tokens_word = sum(token_counts) / sum(word_counts)
    avg_tokens_sentence = sum(token_counts) / len(sentences)

    return avg_tokens_word, avg_tokens_sentence

sentences = ds_euscrawl['train']['text'][:50000]
avg_tokens_word, avg_tokens_sentence = avg_tokens_per_word(sentences)
print(f"Average tokens per word: {avg_tokens_word:.2f}")
print(f"Average tokens per sentence: {avg_tokens_sentence:.2f}")