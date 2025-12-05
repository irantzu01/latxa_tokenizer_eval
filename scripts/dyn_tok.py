from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from zett.utils import get_surface_form_matrix
from datasets import load_dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from collections import OrderedDict


import sys
sys.path.append("../../dynamic-tokenization")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load Latxa model and tokenizer
model = AutoModelForCausalLM.from_pretrained("HiTZ/latxa-7b-v1.2")
latxa_tokenizer = AutoTokenizer.from_pretrained("HiTZ/latxa-7b-v1.2")
print("Latxa model and tokenizer loaded.")


# Load hypernetwork
hypernet = AutoModel.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental",
    trust_remote_code=True
)
hypernet_tokenizer = AutoTokenizer.from_pretrained(
    "benjamin/zett-hypernetwork-Meta-Llama-3-8B-experimental"
)

dynamic_bpe = Dynamic_BPE(
    tokenizer=hypernet_tokenizer,
    tokenizer_boundary="pretokens",
)
print("Hypernetwork + tokenizer + Dynamic BPE ready.")


# Load datasets
ds_EusProficiency = load_dataset("HiTZ/EusProficiency")
ds_EusTrivia = load_dataset("HiTZ/EusTrivia")
ds_EusReading = load_dataset("HiTZ/EusReading")
ds_EusExams = load_dataset("HiTZ/EusExams")
print("Datasets loaded.")


class DynamicAugmenter:
    """
    Runtime augmenter that:
      - takes dynamic tokens (strings) produced per-batch,
      - maps tokens already in latxa_vocab -> keep their ids,
      - for new tokens: allocate new ids, predict embeddings with hypernet,
        and write those embeddings into model's embedding matrix.
    """

    def __init__(self, model, latxa_tokenizer, hypernet, hypernet_tokenizer, cache_limit=50000):
        self.model = model
        self.latxa_tokenizer = latxa_tokenizer
        self.hypernet = hypernet.to(device)
        self.hypernet_tokenizer = hypernet_tokenizer
        # base HF vocab mapping (token string -> id)
        self.vocab = latxa_tokenizer.get_vocab()
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        self.base_vocab_size = len(self.vocab)
        self.cache = OrderedDict()   # token_str -> token_id (preserve insertion order)
        self.cache_embeddings = {}   # token_str -> (in_emb_tensor, out_emb_tensor)
        self.cache_limit = cache_limit
        # Ensure model on device
        self.model.to(device)
        # we will lazily resize embeddings when needed
        self.current_vocab_size = self.base_vocab_size

    def _ensure_capacity(self, n_new):
        """Resize model embeddings to accomodate n_new new ids."""
        new_size = self.current_vocab_size + n_new
        if new_size == self.model.get_input_embeddings().num_embeddings:
            return
        # HF function to resize embeddings; preserves existing weights and creates new rows
        self.model.resize_token_embeddings(new_size)
        self.current_vocab_size = new_size

    def _predict_embeddings_for_tokens(self, tokens_list):
        """
        Use hypernet to predict embeddings for tokens_list (list of token strings).
        Returns dict token -> (pred_in, pred_out) as torch tensors on device.
        """
        # Tokenizer expects list of dicts for get_surface_form_matrix usage
        batch_examples = [{"text": t} for t in tokens_list]

        # Build surface forms matrix (the zett helper expects hypernet_tokenizer)
        surfaces = get_surface_form_matrix(
            [tokens_list],  # pass as list of list? the function in zett returns arrs; adapt if needed
            maxlen=self.hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=self.hypernet_tokenizer
        )[0]  # get first output if returns tuple

        # Build source embeddings matrix from current model (concatenate in/out as in example)
        src_emb = torch.cat([
            self.model.get_input_embeddings().weight.data,
            self.model.get_output_embeddings().weight.data
        ], dim=1).to(device)

        # surfaces -> hypernet prediction (adapt call to hypernet API)
        with torch.no_grad():
            pred_in, pred_out, _ = self.hypernet(
                torch.from_numpy(surfaces).to(device),
                source_embeddings=src_emb
            )

        # pred_in/out shape: (num_tokens, embedding_dim) etc. Convert to CPU/torch tensors
        # Map predicted embeddings to tokens_list order
        result = {}
        for i, t in enumerate(tokens_list):
            result[t] = (pred_in[i].detach().cpu(), pred_out[i].detach().cpu())

        return result
    
    def add_and_assign_new_tokens(self, new_token_strs):
        """
        For token strings not in base vocab and not cached:
           - predict embeddings with hypernet
           - resize model embedding matrix
           - write predicted embeddings to new rows
        Return mapping token_str -> token_id (global)
        """
        # Filter tokens not already in cache or vocab
        to_create = [t for t in new_token_strs if (t not in self.vocab and t not in self.cache)]

        if len(to_create) == 0:
            # build mapping from cache/vocab for requested tokens
            mapping = {}
            for t in new_token_strs:
                if t in self.vocab:
                    mapping[t] = self.vocab[t]
                else:
                    mapping[t] = self.cache[t]
            return mapping

        # Predict embeddings with hypernet in chunks if many
        CHUNK = 128
        predicted = {}
        for i in range(0, len(to_create), CHUNK):
            chunk = to_create[i:i+CHUNK]
            pred_chunk = self._predict_embeddings_for_tokens(chunk)
            predicted.update(pred_chunk)

        # Now allocate ids and ensure capacity
        n_new = len(to_create)
        self._ensure_capacity(n_new)

        # Write embeddings into the model embedding matrix (on CPU then move)
        # We will collect tensors to write to the new rows
        input_emb = self.model.get_input_embeddings().weight.data  # on device
        output_emb = self.model.get_output_embeddings().weight.data

        # assign sequentially at the end
        assigned = {}
        next_id = self.current_vocab_size - n_new  # first index of newly created rows
        # But careful: model.resize_token_embeddings sets current_vocab_size earlier. We stored it there.

        # Actually recompute next_id as base + existing cache size
        next_id = self.base_vocab_size + len([k for k in self.cache]) 

        for t in to_create:
            in_emb_cpu, out_emb_cpu = predicted[t]  # CPU tensors
            in_emb = in_emb_cpu.to(device)
            out_emb = out_emb_cpu.to(device)
            # new id
            new_id = self.base_vocab_size + len(self.cache)
            # Append to cache and embeddings
            self.cache[t] = new_id
            self.cache_embeddings[t] = (in_emb_cpu, out_emb_cpu)
            # assign into model weights
            # Note: input_emb and output_emb are tensors on device; assign by index
            self.model.get_input_embeddings().weight.data[new_id, :] = in_emb
            self.model.get_output_embeddings().weight.data[new_id, :] = out_emb
            assigned[t] = new_id

            # enforce cache limit
            if len(self.cache) > self.cache_limit:
                # pop oldest
                old_token, old_id = self.cache.popitem(last=False)
                self.cache_embeddings.pop(old_token, None)
                # We do not reclaim embedding rows to keep indices stable (complex). Accept growth or restart.

        # Build mapping for all requested tokens (new_token_strs)
        mapping = {}
        for t in new_token_strs:
            if t in self.vocab:
                mapping[t] = self.vocab[t]
            else:
                mapping[t] = self.cache[t]

        # Update current_vocab_size if needed
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings

        return mapping

    def tokens_to_ids(self, tokenized_batch):
        """
        Convert a batch tokenized as lists of token strings (dynamic tokens)
        into lists of token ids (ints) using base vocab + cache.
        tokenized_batch: list[list[str]]
        Returns: list[list[int]]
        """
        # gather all unique tokens that are not in base vocab
        uniques = set(t for seq in tokenized_batch for t in seq)
        new_tokens = [t for t in uniques if t not in self.vocab]
        # ensure they are created/assigned
        mapping = self.add_and_assign_new_tokens(new_tokens)
        # Now map sequences
        out_ids = []
        for seq in tokenized_batch:
            ids = []
            for t in seq:
                if t in self.vocab:
                    ids.append(self.vocab[t])
                else:
                    ids.append(self.cache[t])
            out_ids.append(ids)
        return out_ids


# Utility function to normalize dynamic BPE tokens

def normalize_dynbpe_tokens(batch_tokens):
    cleaned = []
    for seq in batch_tokens:
        new_seq = []
        for tok in seq:
            # remove leading GPT whitespace marker if present
            if tok.startswith("Ġ"):
                tok = tok[1:]

            # if token is multi-character, split into characters
            # because byte tokenizer expects char-level tokens
            for ch in tok:
                new_seq.append(ch)
        cleaned.append(new_seq)
    return cleaned


augmenter = DynamicAugmenter(
    model=model,
    latxa_tokenizer=latxa_tokenizer,
    hypernet=hypernet,
    hypernet_tokenizer=hypernet_tokenizer,
    cache_limit=50000
)

BATCH_SIZE = 64
for i in range(0, len(sentences), BATCH_SIZE):
    batch = sentences[i:i+BATCH_SIZE]
    print(len(batch))
    batch = [s for s in batch if s.strip() != ""]
    print(len(batch))
    examples = [{"text": s, "pretokens": s.split()} for s in batch]

    # 1) Dynamic BPE returns token strings per sentence
    dyn_tokens, _, _, _ = dynamic_bpe.tokenize_batch(
        batch_examples=examples,
        max_nr_merges=30,
        mlm=True
    )
    # dyn_tokens is list[list[str]]
    print("dyn_tokens example:", dyn_tokens[0])
    # Normalize tokens (remove Ġ, split multi-char into chars)
    dyn_tokens = normalize_dynbpe_tokens(dyn_tokens)
    print("Normalized dyn_tokens example:", dyn_tokens[0])

    # 2) Map tokens to ids, creating new embeddings as needed
    batch_ids = augmenter.tokens_to_ids(dyn_tokens)  # list of lists
    print(batch_ids[:2])

    # 3) Convert to padded tensors for model
    # pad with tokenizer.pad_token_id if you have one; else 0
    pad_id = latxa_tokenizer.pad_token_id or latxa_tokenizer.eos_token_id
    maxlen = max(len(x) for x in batch_ids)
    input_ids = torch.full((len(batch_ids), maxlen), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for r, seq in enumerate(batch_ids):
        input_ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[r, :len(seq)] = 1

    # 4) Run the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # ... downstream evaluation ...
    

