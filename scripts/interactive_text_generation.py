from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ========================
# SETTINGS
# ========================
MODEL_NAME = "HiTZ/latxa-7b-v1.1"

# ========================
# LOAD MODEL AND TOKENIZER
# ========================
print("Loading tokenizer and model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")
print("Model loaded on CPU.")

# ========================
# CREATE TEXT-GENERATION PIPELINE
# ========================
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ========================
# INTERACTIVE PROMPT LOOP
# ========================
print("\n=== Latxa Playground ===\n")
while True:
    prompt = input("Enter a Basque prompt (or 'exit' to quit):\n> ")
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    # Show tokenization
    token_ids = tokenizer.encode(prompt)
    print("\nToken IDs:", token_ids)
    print("Decoded back:", tokenizer.decode(token_ids))

    # Ask user for generation parameters
    try:
        max_tokens = int(input("Max new tokens (default 50): ") or 50)
        do_sample = input("Use sampling? (y/n, default y): ").lower() != "n"
        top_k = int(input("Top-k (default 50): ") or 50)
        top_p = float(input("Top-p (default 0.95): ") or 0.95)
        num_return = int(input("Number of outputs (default 3): ") or 3)
    except Exception:
        print("Invalid input, using defaults.")
        max_tokens, do_sample, top_k, top_p, num_return = 50, True, 50, 0.95, 3

    print("\nGenerating text (this may take a while on CPU)...\n")
    outputs = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return
    )

    for i, out in enumerate(outputs):
        print(f"\n--- Option {i+1} ---\n{out['generated_text']}\n")


