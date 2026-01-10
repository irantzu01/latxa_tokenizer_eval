#!/usr/bin/env python3
"""
Sample a subset of the Basque corpus for faster training.
"""

import random
import os

# ================== Settings ==================
input_file = "data/basque_corpus.txt"
output_file = "data/basque_corpus_sampled.txt"
sample_size = 500_000  # Number of lines to sample (adjust as needed)
random_seed = 42

# ================== Reservoir Sampling ==================
print(f"Sampling {sample_size:,} lines from {input_file}...")
print("This uses reservoir sampling to handle large files efficiently.\n")

random.seed(random_seed)
reservoir = []

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        if i < sample_size:
            # Fill reservoir with first sample_size lines
            reservoir.append(line)
        else:
            # Randomly replace elements with decreasing probability
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = line
        
        # Progress indicator
        if (i + 1) % 100_000 == 0:
            print(f"Processed {i+1:,} lines...")

total_lines = i + 1
print(f"\nTotal lines processed: {total_lines:,}")
print(f"Lines sampled: {len(reservoir):,}")
print(f"Sampling ratio: {len(reservoir)/total_lines*100:.2f}%\n")

# ================== Save Sampled Data ==================
print(f"Saving sampled data to {output_file}...")

# Shuffle the reservoir for good measure
random.shuffle(reservoir)

with open(output_file, "w", encoding="utf-8") as f:
    for line in reservoir:
        f.write(line + "\n")

print(f"✓ Done! Sampled corpus saved to {output_file}")

# ================== File Size Info ==================
input_size = os.path.getsize(input_file) / (1024**3)  # GB
output_size = os.path.getsize(output_file) / (1024**3)  # GB

print(f"\nFile sizes:")
print(f"  Original: {input_size:.2f} GB")
print(f"  Sampled:  {output_size:.2f} GB")
print(f"  Space saved: {input_size - output_size:.2f} GB")

# ================== Estimated Training Time ==================
batch_size = 4
gradient_accumulation_steps = 8
epochs = 3

estimated_steps = (sample_size // batch_size) // gradient_accumulation_steps * epochs
estimated_hours = estimated_steps / 1000  # Very rough estimate: ~1000 steps/hour

print(f"\nEstimated training:")
print(f"  Steps per epoch: {sample_size // batch_size:,}")
print(f"  Total steps (3 epochs): {estimated_steps:,}")
print(f"  Estimated time: ~{estimated_hours:.1f} hours")
print(f"  (This is a rough estimate - actual time may vary)")
