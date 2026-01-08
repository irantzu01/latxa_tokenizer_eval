import sentencepiece as spm
from datasets import load_dataset
import os

# Path to the corpus file
corpus_file = "data/basque_corpus.txt"


# Output prefix for the tokenizer files
tokenizer_prefix = "basque_bpe_32k"

# ========== 2. Configure SentencePiece training ==========
spm.SentencePieceTrainer.train(
    input=corpus_file,
    model_prefix=tokenizer_prefix,
    vocab_size=32000,
    character_coverage=1.0,
    model_type='bpe',        # BPE tokenizer
    unk_id=0,                # <unk> token
    pad_id=-1,               # no padding token
    bos_id=1,                # <s> token
    eos_id=2,                # </s> token
    user_defined_symbols=[], # add other special tokens if needed
    byte_fallback=True,      # enable byte fallback
    split_digits=True,
    normalization_rule_name="nfkc",
    train_extremely_large_corpus=True  # helps with large corpora
)

# ========== 3. Test the tokenizer ==========
import sentencepiece as sp

sp_model = sp.SentencePieceProcessor()
sp_model.load(f"{tokenizer_prefix}.model")

test_sentence = "Euskal Herria da gure herria."
tokens = sp_model.encode(test_sentence, out_type=str)
token_ids = sp_model.encode(test_sentence, out_type=int)

print("Tokens:", tokens)
print("Token IDs:", token_ids)
