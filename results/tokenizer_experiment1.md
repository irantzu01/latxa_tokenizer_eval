RESULTS FIRST RUNNING LATXA WITH NEW TOKENIZER
DATASET: SMALL CORPUS, 100K LINES

Vocab overlap: 7420 tokens
New tokens: 24580 tokens
Middle layers frozen. Only new token embeddings + LM head will be trained.
Trainable parameters: 262,144,000 / 6,738,415,616 (3.89%)
Counting lines in data/basque_corpus_sampled_small.txt...
Total lines: 100,000
Train lines: 99,000, Val lines: 1,000
Counting lines in data/basque_corpus_sampled_small.txt...
Total lines: 100,000
Train lines: 99,000, Val lines: 1,000

Estimated steps per epoch: 24,750
Gradient accumulation steps: 8
Effective batch size: 32
Total training steps: 9,279
Warmup steps: 463

==================================================
Starting training...
==================================================


==================================================
Epoch 1/3
==================================================

Epoch 1 average loss: 4.0867

Running validation...
Validation - Loss: 3.1453, Perplexity: 23.23
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/checkpoint-epoch1
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/epoch-1
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Epoch 2/3
==================================================

Epoch 2 average loss: 2.9451

Running validation...
Validation - Loss: 3.1332, Perplexity: 22.95
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/checkpoint-epoch2
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/epoch-2
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/epoch-1
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Epoch 3/3
==================================================

Epoch 3 average loss: 2.8479

Running validation...
Validation - Loss: 3.1405, Perplexity: 23.12
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/epoch-3
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/epoch-2
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Training complete!
Final model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100K/final'
Best validation perplexity: 22.95
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria.

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia, baina, ezta, eta, ezta.

Input: Bilbo hiriak
Generated: Bilbo hiriak, bestela, eta besteta km)