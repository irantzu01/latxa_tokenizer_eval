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
                 cache_limit=50000, device=None):
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(device)
        self.latxa_tokenizer = latxa_tokenizer
        self.hypernet = hypernet.to(device)
        self.hypernet_tokenizer = hypernet_tokenizer
        self.vocab = latxa_tokenizer.get_vocab()
        self.reverse_vocab = {v:k for k,v in self.vocab.items()}
        self.base_vocab_size = len(self.vocab)
        self.cache = OrderedDict()   # token_str -> token_id
        self.cache_embeddings = {}   # token_str -> (in_emb_tensor, out_emb_tensor)
        self.cache_limit = cache_limit
        self.current_vocab_size = self.base_vocab_size
        
        print(f"DynamicAugmenter initialized:")
        print(f"  Base vocab size: {self.base_vocab_size}")
        print(f"  Cache limit: {cache_limit}")
        print(f"  Device: {device}")

    def _ensure_capacity(self, n_new):
        """Resize model embeddings to accomodate n_new new ids."""
        new_size = self.current_vocab_size + n_new
        if new_size > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(new_size)
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings

    def _predict_embeddings_for_tokens(self, tokens_list):
        """
        Predict embeddings for dynamic tokens using Zett hypernetwork.
        Follows the exact approach from the Zett paper.
        Returns dict[token] -> (in_emb, out_emb) on CPU.
        """

        # Debug: check what tokens we're getting
        print(f"DEBUG: Predicting embeddings for {len(tokens_list)} tokens")
        print(f"DEBUG: Sample tokens: {tokens_list[:5]}")
        
        # get_surface_form_matrix expects tokens in byte format (e.g., 'Ġhello')
        # Make sure tokens are properly formatted
        surfaces = get_surface_form_matrix(
            tokens_list,  # Direct list of tokens
            maxlen=self.hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=self.hypernet_tokenizer
        )

        # Unpack tuple - get_surface_form_matrix returns (surfaces, ...)
        if isinstance(surfaces, tuple):
            surfaces = surfaces[0]

        # Convert to tensor on device
        surfaces = torch.from_numpy(surfaces).to(self.device)

        # ==================== BUILD SOURCE EMBEDDINGS ====================
        # Concatenate input + output embeddings as per Zett paper
        src_emb = torch.cat([
            self.model.get_input_embeddings().weight.data,  # [vocab, 4096]
            self.model.get_output_embeddings().weight.data,  # [vocab, 4096]
        ], dim=1).to(self.device)  # [vocab, 8192]

        with torch.no_grad():
            try:
                # Predict embeddings using hypernet
                pred_in, pred_out, _ = self.hypernet(
                    surfaces,
                    source_embeddings=src_emb
                )
                # pred_in: [batch, 4096] - input embeddings
                # pred_out: [batch, 4096] - output embeddings
            except RuntimeError as e:
                print(f"ERROR in hypernet call:")
                print(f"  surfaces shape: {surfaces.shape}")
                print(f"  src_emb shape: {src_emb.shape}")
                print(f"  surfaces dtype: {surfaces.dtype}, device: {surfaces.device}")
                print(f"  surfaces min/max: {surfaces.min()}/{surfaces.max()}")
                print(f"  tokens_list sample: {tokens_list[:5]}")
                raise e

        # Return CPU tensors as dict
        result = {}
        for i, tok in enumerate(tokens_list):
            result[tok] = (
                pred_in[i].detach().cpu(),
                pred_out[i].detach().cpu(),
            )

        return result

    def add_and_assign_new_tokens(self, new_token_strs):
        """Add new dynamic tokens to cache, predict embeddings, and assign IDs."""
        # Filter tokens not in vocab/cache (no normalization here)
        to_create = [t for t in new_token_strs if t not in self.vocab and t not in self.cache]
        if not to_create:
            # Build mapping from existing tokens
            mapping = {t: self.vocab.get(t, self.cache.get(t)) for t in new_token_strs}
            return mapping

        print(f"Creating {len(to_create)} new tokens...")

        # Predict embeddings in chunks (keeping original token format)
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
        mapping = {t: self.vocab.get(t, self.cache.get(t)) for t in new_token_strs}
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings
        
        print(f"✓ Created {len(to_create)} new tokens. Total dynamic tokens: {len(self.cache)}")
        
        return mapping

    def tokens_to_ids(self, tokenized_batch):
        """Convert batch of token strings to token IDs using vocab + dynamic cache."""
        uniques = set(t for seq in tokenized_batch for t in seq)
        
        # IMPORTANT: Don't normalize tokens before checking vocab
        # We need to keep the original format (with Ġ, ▁, etc.) for the hypernet
        new_tokens = [t for t in uniques if t not in self.vocab and t not in self.cache]
        
        # Add new tokens if needed
        if new_tokens:
            mapping = self.add_and_assign_new_tokens(new_tokens)
        
        # Convert sequences to IDs
        out_ids = []
        for seq in tokenized_batch:
            ids = []
            for t in seq:
                if t in self.vocab:
                    ids.append(self.vocab[t])
                elif t in self.cache:
                    ids.append(self.cache[t])
                else:
                    # This shouldn't happen, but handle it
                    print(f"WARNING: Token '{t}' not in vocab or cache!")
                    # Try to find it after normalization as fallback
                    normalized = normalize_dynamic_token(t)
                    if normalized in self.cache:
                        ids.append(self.cache[normalized])
                    else:
                        # Use unk token
                        ids.append(self.vocab.get('<unk>', 0))
            out_ids.append(ids)
        return out_ids


# ==================== HELPER FUNCTIONS ====================
def normalize_dynamic_token(tok: str) -> str:
    """Normalize token by removing special characters (use only as fallback)."""
    tok = tok.replace("Ġ", "").replace("▁", "").replace("<s>", "").replace("</s>", "").strip()
    if tok == "":
        tok = " "
    return tok

def expand_to_char_tokens(tokens):
    """Expand tokens to character lists."""
    return [list(normalize_dynamic_token(t)) for t in tokens]