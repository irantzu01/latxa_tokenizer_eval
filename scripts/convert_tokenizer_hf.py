#!/usr/bin/env python3
"""
Convert a trained SentencePiece tokenizer to Hugging Face format
for use with Transformers models (e.g., LLaMA).
"""

import os
from transformers import PreTrainedTokenizerFast

# ================== 1. Paths ==================
sp_model_file = "basque_bpe_32k.model"      # Your trained SentencePiece model
output_dir = "basque_tokenizer_hf"         # Where the HF tokenizer will be saved

os.makedirs(output_dir, exist_ok=True)

# ================== 2. Create HF tokenizer ==================
# PreTrainedTokenizerFast wraps a SentencePiece model
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=None,            # Not needed when using sp_model
    model_max_length=2048,          # Set according to your model
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>"
)

# Attach the SentencePiece model
tokenizer._tokenizer.model = sp_model_file  # Link the SP model
tokenizer.add_special_tokens({
    "unk_token": "<unk>",
    "bos_token": "<s>",
    "eos_token": "</s>"
})

# ================== 3. Save HF tokenizer ==================
tokenizer.save_pretrained(output_dir)
print(f"Hugging Face tokenizer saved in '{output_dir}'")

# ================== 4. Test the tokenizer ==================
test_sentence = "Euskal Herria da gure herria."
tokens = tokenizer.tokenize(test_sentence)
token_ids = tokenizer(test_sentence)['input_ids']

print("Test sentence:", test_sentence)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
