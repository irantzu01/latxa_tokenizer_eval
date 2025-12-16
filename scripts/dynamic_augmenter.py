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
import torch
import numpy as np



class DynamicAugmenter:
    """
    Runtime augmenter that:
      - takes dynamic tokens (strings) produced per-batch,
      - maps tokens already in latxa_vocab -> keep their ids,
      - for new tokens: allocate new ids, predict embeddings with hypernet,
        and write those embeddings into model's embedding matrix.
    """

    def __init__(self, model, latxa_tokenizer, hypernet, hypernet_tokenizer,
                 source_model, cache_limit=50000, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.latxa_tokenizer = latxa_tokenizer
        self.hypernet = hypernet.to(self.device)
        self.hypernet_tokenizer = hypernet_tokenizer
        self.source_model = source_model
        self.vocab = latxa_tokenizer.get_vocab()
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        self.base_vocab_size = len(self.vocab)
        self.cache = OrderedDict()   # token_str -> token_id
        self.cache_embeddings = {}   # token_str -> (in_emb_tensor, out_emb_tensor)
        self.cache_limit = cache_limit
        self.current_vocab_size = self.base_vocab_size

    def _ensure_capacity(self, n_new):
        """Resize model embeddings to accomodate n_new new ids."""
        new_size = self.current_vocab_size + n_new
        if new_size > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(new_size)
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings

    def _predict_embeddings_for_tokens(self, tokens_list):
        """
        Use hypernet to predict embeddings for tokens_list (list of token strings).
        Returns dict token -> (pred_in, pred_out) as torch tensors on self.device.
        """
        # Convert tokens to characters
        char_tokens = expand_to_char_tokens(tokens_list)
        char_strings = ["".join(chars) for chars in char_tokens]

        # Get surface form matrix from Zett
        surfaces = get_surface_form_matrix(
            char_strings,
            maxlen=self.hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=self.hypernet_tokenizer
        )

        # Handle tuple or array
        if isinstance(surfaces, tuple):
            surfaces = surfaces[0]
        if isinstance(surfaces, np.ndarray):
            surfaces = torch.from_numpy(surfaces)
        elif not isinstance(surfaces, torch.Tensor):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(surfaces)}")

        surfaces = surfaces.to(self.device)

        # Build source embeddings matrix from source model
        src_emb = torch.cat([
            self.source_model.get_input_embeddings().weight,
            self.source_model.get_output_embeddings().weight,],
        dim=1).to(device)

        # Predict embeddings with hypernet
        with torch.no_grad():
            pred_in, pred_out, _ = self.hypernet(surfaces)

        result = {}
        for i, t in enumerate(tokens_list):
            result[t] = (pred_in[i].detach(), pred_out[i].detach())

        return result

    def add_and_assign_new_tokens(self, new_token_strs):
        """Add new dynamic tokens to cache, predict embeddings, and assign IDs."""
        # Filter tokens not in vocab/cache
        to_create = [t for t in new_token_strs if t not in self.vocab and t not in self.cache]
        if not to_create:
            # Build mapping from existing tokens
            mapping = {t: self.vocab.get(t, self.cache[t]) for t in new_token_strs}
            return mapping

        # Predict embeddings in chunks
        CHUNK = 128
        predicted = {}
        for i in range(0, len(to_create), CHUNK):
            chunk = to_create[i:i+CHUNK]
            pred_chunk = self._predict_embeddings_for_tokens(chunk)
            predicted.update(pred_chunk)

        # Resize embeddings
        self._ensure_capacity(len(to_create))

        # Assign new IDs and embeddings
        for t in to_create:
            new_id = self.base_vocab_size + len(self.cache)
            in_emb, out_emb = predicted[t]
            in_emb = in_emb.to(self.device)
            out_emb = out_emb.to(self.device)

            self.cache[t] = new_id
            self.cache_embeddings[t] = (in_emb, out_emb)
            self.model.get_input_embeddings().weight.data[new_id, :] = in_emb
            self.model.get_output_embeddings().weight.data[new_id, :] = out_emb

            # enforce cache limit
            if len(self.cache) > self.cache_limit:
                old_token, _ = self.cache.popitem(last=False)
                self.cache_embeddings.pop(old_token, None)

        # Build mapping
        mapping = {t: self.vocab.get(t, self.cache[t]) for t in new_token_strs}
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings
        return mapping

    def tokens_to_ids(self, tokenized_batch):
        """Convert batch of token strings to token IDs using vocab + dynamic cache."""
        uniques = set(t for seq in tokenized_batch for t in seq)
        new_tokens = [normalize_dynamic_token(t) for t in uniques if t not in self.vocab]
        # Normalize dynamic tokens
        mapping = self.add_and_assign_new_tokens(new_tokens)
        out_ids = []
        for seq in tokenized_batch:
            ids = [self.vocab.get(normalize_dynamic_token(t), self.cache[t]) for t in seq]
            out_ids.append(ids)
        return out_ids


# Helper functions
def normalize_dynamic_token(tok: str) -> str:
    tok = tok.replace("Ġ", "").replace("▁", "").replace("<s>", "").replace("</s>", "").strip()
    if tok == "":
        tok = " "
    return tok

def expand_to_char_tokens(tokens):
    return [list(normalize_dynamic_token(t)) for t in tokens]