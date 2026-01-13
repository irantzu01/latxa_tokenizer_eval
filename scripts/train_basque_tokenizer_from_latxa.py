#!/usr/bin/env python3
"""
Train a new Basque tokenizer from Latxa's tokenizer.
Uses train_new_from_iterator to automatically inherit all settings.
Only the vocabulary will change based on the Basque corpus.
"""

from transformers import AutoTokenizer
from tqdm import tqdm
import os

# ==================== CONFIGURATION ====================
original_tokenizer_name = "HiTZ/latxa-7b-v1.2"
corpus_file = "data/basque_corpus.txt"
output_dir = "basque_tokenizer_from_latxa"
vocab_size = 32000

print(f"\n{'='*60}")
print(f"Training New Basque Tokenizer from Latxa")
print(f"{'='*60}")
print(f"Original tokenizer: {original_tokenizer_name}")
print(f"Corpus file: {corpus_file}")
print(f"Output directory: {output_dir}")
print(f"Vocabulary size: {vocab_size}")
print(f"{'='*60}\n")

# ==================== LOAD ORIGINAL TOKENIZER ====================
print("Loading original Latxa tokenizer...")
original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_name)
print(f"✓ Loaded Latxa tokenizer")
print(f"  Original vocab size: {len(original_tokenizer)}")
print(f"  Tokenizer type: {type(original_tokenizer).__name__}")
print(f"  Special tokens: {original_tokenizer.all_special_tokens}")

# ==================== PREPARE TRAINING CORPUS ====================
def get_training_corpus(batch_size=1000):
    """
    Generator that yields batches of text from the corpus.
    Batching improves training speed.
    """
    print("\nReading corpus...")
    
    # Count lines for progress bar
    with open(corpus_file, "r", encoding="utf-8") as f:
        num_lines = sum(1 for _ in f)
    
    print(f"Total lines in corpus: {num_lines:,}")
    
    batch = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=num_lines, desc="Processing corpus"):
            line = line.strip()
            if line:  # Skip empty lines
                batch.append(line)
                
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        
        # Yield remaining lines
        if batch:
            yield batch

# ==================== TRAIN NEW TOKENIZER ====================
print("\nTraining new tokenizer on Basque corpus...")
print("This will inherit all settings from Latxa and only retrain the vocabulary.")
print("This may take several minutes...\n")

new_tokenizer = original_tokenizer.train_new_from_iterator(
    get_training_corpus(batch_size=1000),
    vocab_size=vocab_size
)

print(f"✓ Training complete!")
print(f"  New vocab size: {len(new_tokenizer)}")

# ==================== VERIFY SPECIAL TOKENS ====================
print("\nVerifying special tokens...")
print(f"  Special tokens: {new_tokenizer.all_special_tokens}")
print(f"  BOS token: {new_tokenizer.bos_token} (ID: {new_tokenizer.bos_token_id})")
print(f"  EOS token: {new_tokenizer.eos_token} (ID: {new_tokenizer.eos_token_id})")
print(f"  UNK token: {new_tokenizer.unk_token} (ID: {new_tokenizer.unk_token_id})")
print(f"  PAD token: {new_tokenizer.pad_token} (ID: {new_tokenizer.pad_token_id})")

# ==================== SAVE NEW TOKENIZER ====================
os.makedirs(output_dir, exist_ok=True)
new_tokenizer.save_pretrained(output_dir)
print(f"\n✓ New tokenizer saved to '{output_dir}'")

# ==================== COMPARE VOCABULARIES ====================
print("\n" + "="*60)
print("Vocabulary Comparison")
print("="*60)

original_vocab = set(original_tokenizer.get_vocab().keys())
new_vocab = set(new_tokenizer.get_vocab().keys())

# Calculate overlap
overlap = original_vocab & new_vocab
only_original = original_vocab - new_vocab
only_new = new_vocab - original_vocab

print(f"Original vocab: {len(original_vocab):,} tokens")
print(f"New vocab: {len(new_vocab):,} tokens")
print(f"Overlap: {len(overlap):,} tokens ({100*len(overlap)/len(original_vocab):.1f}%)")
print(f"Only in original: {len(only_original):,} tokens")
print(f"Only in new: {len(only_new):,} tokens")

# Show some examples of new tokens
if only_new:
    print(f"\nSample new Basque-specific tokens (first 20):")
    for i, token in enumerate(sorted(only_new)[:20]):
        print(f"  {token}")

# ==================== TEST TOKENIZATION ====================
print("\n" + "="*60)
print("Testing Tokenization")
print("="*60)

test_sentences = [
    "Euskal Herria da gure herria.",
    "Gaur egun, teknologia aurreratua dugu.",
    "Bilboko hirian bizi naiz.",
]

for sentence in test_sentences:
    print(f"\nSentence: {sentence}")
    
    # Original tokenizer
    original_tokens = original_tokenizer.tokenize(sentence)
    print(f"  Original ({len(original_tokens)} tokens): {original_tokens}")
    
    # New tokenizer
    new_tokens = new_tokenizer.tokenize(sentence)
    print(f"  New ({len(new_tokens)} tokens): {new_tokens}")
    
    # Compare
    if len(new_tokens) < len(original_tokens):
        print(f"  ✓ New tokenizer is more efficient ({len(original_tokens) - len(new_tokens)} fewer tokens)")
    elif len(new_tokens) > len(original_tokens):
        print(f"  ⚠ New tokenizer uses more tokens (+{len(new_tokens) - len(original_tokens)})")
    else:
        print(f"  = Same number of tokens")

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"✓ New Basque tokenizer trained successfully")
print(f"✓ Saved to: {output_dir}")
print(f"✓ Vocabulary size: {len(new_tokenizer):,}")
print(f"✓ Inherits all settings from Latxa tokenizer")
print(f"✓ Ready to use for lexical realignment")
print("="*60 + "\n")

print("Next steps:")
print("1. Use this tokenizer for lexical realignment with your training script")
print("2. Compare performance with your SentencePiece-trained tokenizer")
print(f"3. Load with: tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")