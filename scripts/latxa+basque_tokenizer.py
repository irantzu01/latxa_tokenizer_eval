from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# ================== 1️⃣ Paths ==================
model_name = "HiTZ/latxa-7b-v1.2"   # Latxa 7B
tokenizer_dir = "basque_tokenizer_hf"  # Your HF tokenizer folder
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 2️⃣ Load the Basque tokenizer ==================
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)

# ================== 3️⃣ Load the Latxa model ==================
# Set low_cpu_mem_usage=True if you are tight on GPU memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ================== 4️⃣ Lexical realignment ==================
# This expands the embedding table to match new tokenizer vocab
# Only necessary if your tokenizer has tokens that the model does not
old_vocab_size = model.get_input_embeddings().weight.size(0)
new_vocab_size = len(tokenizer)
print(f"Old vocab size: {old_vocab_size}, New vocab size: {new_vocab_size}")

if new_vocab_size > old_vocab_size:
    model.resize_token_embeddings(new_vocab_size)
    print("Model embeddings resized to match tokenizer!")

# ================== 5️⃣ Test encoding and generation ==================
test_sentence = "Euskal Herria da gure herria."
inputs = tokenizer(test_sentence, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated:", decoded)

# ================== 6️⃣ Save aligned model ==================
save_dir = "latxa7b_basque_aligned"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Aligned model + tokenizer saved to {save_dir}")
