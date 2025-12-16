import sys
import os

project_root = os.path.expanduser(
    "~/MASTER/WiSe25/Lab Rotation/dynamic-tokenization"
)
sys.path.append(project_root)

from collections import OrderedDict
from tokenizations.dynamic_bpe import Dynamic_BPE
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from zett.utils import get_surface_form_matrix



class DynamicAugmenter:
    """
    Runtime augmenter that:
      - takes dynamic tokens (strings) produced per-batch,
      - maps tokens already in latxa_vocab -> keep their ids,
      - for new tokens: allocate new ids, predict embeddings with hypernet,
        and write those embeddings into model's embedding matrix.
    """

    def __init__(self, model, latxa_tokenizer, hypernet, hypernet_tokenizer, cache_limit=50000, device='cpu'):
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
        char_tokens = expand_to_char_tokens(tokens_list)

        assert all(
        isinstance(c, str) and len(c) == 1
        for token in char_tokens
        for c in token
        )   

        surfaces = get_surface_form_matrix(
            char_tokens,
            maxlen=self.hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=self.hypernet_tokenizer
        )

        # # Build surface forms matrix (the zett helper expects hypernet_tokenizer)
        # surfaces = get_surface_form_matrix(
        #     [tokens_list],  # pass as list of list? the function in zett returns arrs; adapt if needed
        #     maxlen=self.hypernet.config.hn_surface_maxlen,
        #     tokenizer_to_use=self.hypernet_tokenizer
        # )[0]  # get first output if returns tuple

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
        new_tokens = [normalize_dynamic_token(t) for t in new_tokens]
        #Make sure to normalize dynamic tokens
        bad = [t for t in new_tokens if "Ġ" in t or "▁" in t]
        print("Bad tokens:", bad[:10])
        # ensure they are created/assigned
        mapping = self.add_and_assign_new_tokens(new_tokens)
        # Now map sequences
        out_ids = []
        for seq in tokenized_batch:
            ids = []
            for t in seq:
                t = normalize_dynamic_token(t)
                if t in self.vocab:
                    ids.append(self.vocab[t])
                else:
                    ids.append(self.cache[t])
            out_ids.append(ids)
        return out_ids
    


def normalize_dynamic_token(tok: str) -> str:
    """
    Normalize dynamic tokens to plain surface forms
    compatible with Zett hypernetwork.
    """
    # Common whitespace / BPE markers
    tok = tok.replace("Ġ", "")
    tok = tok.replace("▁", "")
    tok = tok.replace("<s>", "")
    tok = tok.replace("</s>", "")
    tok = tok.strip()

    # Avoid empty tokens
    if tok == "":
        tok = " "

    return tok


def expand_to_char_tokens(tokens):
    expanded = []
    for t in tokens:
        t = normalize_dynamic_token(t)
        expanded.append(list(t))  # ['a','l','a','b','a']
    return expanded