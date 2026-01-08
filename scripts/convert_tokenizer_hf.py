#!/usr/bin/env python3
"""
Convert a trained SentencePiece tokenizer to Hugging Face format (slow tokenizer)
for use with Transformers models like LLaMA.
"""

from transformers import LlamaTokenizer
import os

# ========== 1. Paths ==========
sp_model_file = "basque_bpe_32k.model"   # Your trained SentencePiece model
output_dir = "basque_tokenizer_hf"
os.makedirs(output_dir, exist_ok=True)

# ========== 2. Load SentencePiece model into LlamaTokenizer ==========
tokenizer = LlamaTokenizer(
    sp_model_file,
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>"
)

# ========== 3. Save Hugging Face tokenizer ==========
tokenizer.save_pretrained(output_dir)
print(f"Hugging Face tokenizer saved in '{output_dir}'")

# ========== 4. Test ==========
test_sentence = "Euskal Herria da gure herria."
tokens = tokenizer.tokenize(test_sentence)
token_ids = tokenizer(test_sentence)['input_ids']

print("Test sentence:", test_sentence)
print("Tokens:", tokens)
print("Token IDs:", token_ids)

