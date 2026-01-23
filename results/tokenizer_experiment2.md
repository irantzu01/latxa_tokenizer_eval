tokenizer experiments 2


============================================================
Step 4: Embedding Alignment with FOCUS
============================================================

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

Epoch 1 average loss: 1.6531

Running validation...
Validation - Loss: 1.2250, Perplexity: 3.40
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved/final (Epoch 1, PPL: 3.40)

==================================================
Epoch 2/3
==================================================

Epoch 2 average loss: 0.9731

Running validation...
Validation - Loss: 1.1437, Perplexity: 3.14
Removing previous best checkpoint...
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved/final (Epoch 2, PPL: 3.14)

==================================================
Epoch 3/3
==================================================

Epoch 3 average loss: 0.8128

Running validation...
Validation - Loss: 1.1542, Perplexity: 3.17
  No improvement (best PPL: 3.14 from Epoch 2)

==================================================
Training complete!
Best model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_FOCUS_improved/final'
Best validation perplexity: 3.14 (Epoch 2)
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria.

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia digitala erabiltzen da, hala nola ordenagailu bidezko simulazioak eta sare sozialak. Esate baterako, 2010eko urrian "The Sims"

Input: Bilbo hiriak
Generated: Bilbo hiriak eta Bilboko metropoliak osatzen dute. Bi hiriburuetatik, Bilbo eta Bilbotik, hain zuzen ere, Bilbotik eta Bilbotik abiatzen dira bi autobide nagusiak







============================================================
Training Configuration
============================================================
Corpus file: data/basque_corpus_sampled_250k.txt
Output directory: /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_improved
Batch size: 8
Gradient accumulation: 4
Effective batch size: 32
Learning rate: 0.0005
============================================================

Loading tokenizer...
Pad token set to: </s> (2)
Loading Latxa 7B model...
Aligning embeddings with Basque tokenizer...
Vocab overlap: 7420 tokens
New tokens: 24580 tokens
Copying 7420 overlapping token embeddings...
Initializing 24580 new embeddings with mean from existing tokens...
Initializing LM head (output embeddings)...
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

Epoch 1 average loss: 1.8656

Running validation...
Validation - Loss: 1.3657, Perplexity: 3.92
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_improved/final (Epoch 1, PPL: 3.92)

==================================================
Epoch 2/3
==================================================

Epoch 2 average loss: 1.1120

Running validation...
Validation - Loss: 1.2493, Perplexity: 3.49
Removing previous best checkpoint...
✓ New best model saved to /user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_improved/final (Epoch 2, PPL: 3.49)

==================================================
Epoch 3/3
==================================================

Epoch 3 average loss: 0.9377

Running validation...
Validation - Loss: 1.2494, Perplexity: 3.49
  No improvement (best PPL: 3.49 from Epoch 2)

==================================================
Training complete!
Best model saved to '/user/i.elcoroalberdi/u25035/tmp/models/latxa7b_basque_aligned_250k_improved/final'
Best validation perplexity: 3.49 (Epoch 2)
==================================================

Testing generation...

Input: Euskal Herria da gure herria.
Generated: Euskal Herria da gure herria. Nafarroak, Bizkaiek eta Zuberoako, Zuberoako Zuberoako Euskal Herria!

Input: Gaur egun, teknologia
Generated: Gaur egun, teknologia berrien garapena eta informazioaren erabilerak, beste motatako produktu eta zerbitzuen erabilerak, eta abar, kontsumitzaileen premia eta premiak aldatu dituzte. Adibidez, orain ez dago

Input: Bilbo hiriak
Generated: Bilbo hiriak 190.000 bizilagun ditu. Metro sareak 12 linea ditu, 171,6 kilometro luzetakoak, eta
Job finished.