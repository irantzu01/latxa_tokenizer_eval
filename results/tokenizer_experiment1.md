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


Second experiment:
Epoch 3 average loss: 1.3807

Running validation...
Validation - Loss: 1.8025, Perplexity: 6.06
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100k_improved/epoch-3
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100k_improved/epoch-2
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Training complete!
Final model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_100k_improved/final'
Best validation perplexity: 6.05
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria.

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia ezberdin daude:

Input: Bilbo hiriak
Generated: Bilbo hiriak 300 lagunek, 30 milioi inguru daude.
Job finished.

MUCH BETTER!!

BIGGER MODEL:
did not finish, presumably due to storage constraints.
Epoch 1 average loss: 2.0546

Running validation...
Validation - Loss: 1.5217, Perplexity: 4.58
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_500k_improved/checkpoint-epoch1
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_500k_improved/epoch-1
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Epoch 2/3
==================================================

Epoch 2 average loss: 1.5041

MEDIUM MODEL: 250K
Epoch 3 average loss: 1.3967

Running validation...
Validation - Loss: 1.5936, Perplexity: 4.92
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k/epoch-3
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k/epoch-2
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Training complete!
Final model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k/final'
Best validation perplexity: 4.91
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria. 2002ko abenduaren 28an.

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia eta, hizkuntza politika eta gizarte-erretratu baten bila dabiltzala izan zen.

Input: Bilbo hiriak
Generated: Bilbo hiriak (2007).
Job finished.




EXPERIMENT 3: USE FOCUS INITIALIZATION
============================================================
FOCUS Embedding Initialization
============================================================
Processing 32,000 tokens...

Initialization Statistics:
  Exact matches: 7,420 (23.2%)
  Subword composition: 24,580 (76.8%)
  Similarity matching: 0 (0.0%)
  Fallback (mean+noise): 0 (0.0%)

Total new tokens: 24,580
Total old tokens: 7,420

✓ Embeddings initialized with FOCUS
  Vocab size: 32,000
  Old tokens: 7,420
  New tokens: 24,580
Middle layers frozen. All token embeddings + LM head will be trained.
Trainable parameters: 262,144,000 / 6,738,415,616 (3.89%)
Counting lines in data/basque_corpus_sampled_250k.txt...
Total lines: 250,000
Train lines: 247,500, Val lines: 2,500
Counting lines in data/basque_corpus_sampled_250k.txt...
Total lines: 250,000
Train lines: 247,500, Val lines: 2,500

Estimated steps per epoch: 30,937
Gradient accumulation steps: 4
Effective batch size: 32
Total training steps: 23,202
Warmup steps: 2,320

==================================================
Starting training...
==================================================


==================================================
Epoch 1/3
==================================================

Epoch 1 average loss: 1.6524

Running validation...
Validation - Loss: 1.2212, Perplexity: 3.39
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/checkpoint-epoch1
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/epoch-1
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Epoch 2/3
==================================================

Epoch 2 average loss: 0.9706

Running validation...
Validation - Loss: 1.1476, Perplexity: 3.15
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/checkpoint-epoch2
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/epoch-2
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/epoch-1
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Epoch 3/3
==================================================

Epoch 3 average loss: 0.8119

Running validation...
Validation - Loss: 1.1579, Perplexity: 3.18
✓ Checkpoint saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/epoch-3
Removing old checkpoint: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/epoch-2
✓ Old checkpoints cleaned up (keeping last 1)

==================================================
Training complete!
Final model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS/final'
Best validation perplexity: 3.15
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria. Gure herriaren etorkizuna erabakitzeko eskubidea dugu. Eta horretarako eskubidea gauzatzea aldarrikatzen dugu. Gure herriaren etorkizuna erabakitzeko eskubidea dugu. Eta horretarako eskubidea aldarrikatzen dugu. Hori horrela izanik

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia berriak eta informazioaren teknologia (IT) geroz eta garrantzitsuagoak dira hezkuntzan. Helburu nagusia ikasleek gai izan daitezen ikastea eta haien gaitasunak hobetzen laguntzea da. Irakas

Input: Bilbo hiriak
Generated: Bilbo hiriak, Bizkaiko hiriburua, 355.000 bizilagun inguru ditu gaur egun. Udalerri horrek 10 barruti ditu, eta horien barruan,
Job finished.

