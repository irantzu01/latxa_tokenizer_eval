from datasets import load_dataset, concatenate_datasets

# Load datasets
ds1 = load_dataset("HiTZ/latxa-corpus-v1.1", "hplt-v1", split="train")
ds2 = load_dataset("HiTZ/latxa-corpus-v1.1", "wikipedia", split="train")
ds3 = load_dataset("HiTZ/latxa-corpus-v1.1", "egunkaria", split="train")

# Downsample HPLT to ~10%
# Hugging Face returns a dict, so test_size=0.1 keeps 10%
ds1_sampled = ds1.train_test_split(test_size=0.1, seed=42)['test']

# Concatenate datasets
combined_ds = concatenate_datasets([ds1_sampled, ds2, ds3])

# Save text to a file
corpus_file = "data/basque_corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for example in combined_ds:
        line = example['text'].strip()
        if line:
            f.write(line + "\n")
print(f"Corpus saved to {corpus_file}, size: {len(combined_ds)} examples")