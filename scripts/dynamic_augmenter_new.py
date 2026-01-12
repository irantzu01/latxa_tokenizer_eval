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
        
        # ==================== BUILD ALIGNED EMBEDDING MATRIX ====================
        print("Building aligned embedding matrix for hypernet...")
        self._build_aligned_embeddings()
        
        print(f"DynamicAugmenter initialized:")
        print(f"  Base vocab size: {self.base_vocab_size}")
        print(f"  Cache limit: {cache_limit}")
        print(f"  Device: {device}")
    
    def _build_aligned_embeddings(self):
        """
        Build an aligned embedding matrix that covers the hypernet's vocabulary,
        using Latxa's embeddings where tokens overlap.
        """
        # Get vocabulary sizes
        hypernet_vocab_size = len(self.hypernet_tokenizer)
        latxa_vocab_size = len(self.latxa_tokenizer)
        
        print(f"  Latxa vocab size: {latxa_vocab_size}")
        print(f"  Hypernet vocab size: {hypernet_vocab_size}")
        
        # Get Latxa embeddings
        latxa_in_emb = self.model.get_input_embeddings().weight.data  # [32k, 4096]
        latxa_out_emb = self.model.get_output_embeddings().weight.data  # [32k, 4096]
        
        # Concatenate for hypernet format
        latxa_concat = torch.cat([latxa_in_emb, latxa_out_emb], dim=1)  # [32k, 8192]
        
        # Create aligned matrix for hypernet vocab
        self.aligned_embeddings = torch.zeros(
            hypernet_vocab_size,
            8192,
            dtype=latxa_concat.dtype,
            device=self.device
        )
        
        # Map overlapping tokens
        latxa_vocab = self.latxa_tokenizer.get_vocab()
        hypernet_vocab = self.hypernet_tokenizer.get_vocab()
        
        overlap_count = 0
        for token_str, latxa_id in latxa_vocab.items():
            if token_str in hypernet_vocab:
                hypernet_id = hypernet_vocab[token_str]
                if hypernet_id < hypernet_vocab_size:
                    self.aligned_embeddings[hypernet_id] = latxa_concat[latxa_id]
                    overlap_count += 1
        
        # For non-overlapping tokens, initialize with small random values
        # This represents tokens in hypernet vocab but not in Latxa
        unmapped_mask = (self.aligned_embeddings.abs().sum(dim=1) < 1e-6)
        num_unmapped = unmapped_mask.sum().item()
        
        if num_unmapped > 0:
            # Use mean and std from Latxa embeddings for better initialization
            mean_emb = latxa_concat.mean(dim=0)
            std_emb = latxa_concat.std(dim=0).mean().item()
            
            self.aligned_embeddings[unmapped_mask] = (
                mean_emb.unsqueeze(0) + 
                torch.randn(num_unmapped, 8192, dtype=latxa_concat.dtype, device=self.device) * std_emb * 0.02
            )
        
        print(f"  Vocabulary overlap: {overlap_count}/{latxa_vocab_size} tokens ({100*overlap_count/latxa_vocab_size:.1f}%)")
        print(f"  Unmapped tokens: {num_unmapped} (initialized randomly)")
        print(f"✓ Aligned embeddings ready")

    def _ensure_capacity(self, n_new):
        """Resize model embeddings to accomodate n_new new ids."""
        new_size = self.current_vocab_size + n_new
        if new_size > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(new_size)
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings

    def _predict_embeddings_for_tokens(self, tokens_list):
        """
        Predict embeddings for dynamic tokens using Zett hypernetwork.
        Uses pre-built aligned embeddings that include Latxa's context.
        Returns dict[token] -> (in_emb, out_emb) on CPU.
        """

        # Debug: check what tokens we're getting
        print(f"DEBUG: Predicting embeddings for {len(tokens_list)} tokens")
        print(f"DEBUG: Sample tokens: {tokens_list[:5]}")
        
        # get_surface_form_matrix expects tokens in byte format (e.g., 'Ġhello')
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

        # ==================== USE ALIGNED EMBEDDINGS ====================
        # This matrix covers hypernet vocab but uses Latxa embeddings where possible
        
        with torch.no_grad():
            try:
                # Call hypernet WITH aligned source embeddings
                # This gives the hypernet Latxa's context for generating new embeddings
                pred_in, pred_out, _ = self.hypernet(
                    surfaces,
                    source_embeddings=self.aligned_embeddings
                )
                # pred_in: [batch, 4096] - input embeddings for Latxa
                # pred_out: [batch, 4096] - output embeddings for Latxa
            except RuntimeError as e:
                print(f"ERROR in hypernet call:")
                print(f"  surfaces shape: {surfaces.shape}")
                print(f"  aligned_embeddings shape: {self.aligned_embeddings.shape}")
                print(f"  surfaces dtype: {surfaces.dtype}, device: {surfaces.device}")
                if surfaces.numel() > 0:
                    print(f"  surfaces min/max: {surfaces.min().item()}/{surfaces.max().item()}")
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