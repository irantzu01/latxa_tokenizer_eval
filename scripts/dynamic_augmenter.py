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
import torch.nn as nn


# ==================== PROJECTION ADAPTER ====================
class LatxaToLlama3Adapter(nn.Module):
    """
    Adapter to bridge Latxa (8192 hidden dim) to Llama 3 hypernet (4096 hidden dim).
    """
    def __init__(self, latxa_hidden_size=8192, llama3_hidden_size=4096):
        super().__init__()
        self.latxa_hidden_size = latxa_hidden_size
        self.llama3_hidden_size = llama3_hidden_size
        
        # Project Latxa embeddings DOWN to Llama 3 size (for hypernet input)
        self.down_projection = nn.Linear(latxa_hidden_size, llama3_hidden_size, bias=False)
        
        # Project Llama 3 embeddings UP to Latxa size (for hypernet output)
        self.up_projection = nn.Linear(llama3_hidden_size, latxa_hidden_size, bias=False)
        
        # Initialize projections
        self._initialize_projections()
    
    def _initialize_projections(self):
        """Initialize projections intelligently."""
        with torch.no_grad():
            # Down projection: average pairs of dimensions
            down_weight = torch.zeros(self.llama3_hidden_size, self.latxa_hidden_size)
            for i in range(self.llama3_hidden_size):
                down_weight[i, 2*i] = 0.5
                down_weight[i, 2*i + 1] = 0.5
            self.down_projection.weight.data = down_weight
            
            # Up projection: copy first half, then copy again for second half
            up_weight = torch.zeros(self.latxa_hidden_size, self.llama3_hidden_size)
            up_weight[:self.llama3_hidden_size, :] = torch.eye(self.llama3_hidden_size)
            up_weight[self.llama3_hidden_size:, :] = torch.eye(self.llama3_hidden_size)
            self.up_projection.weight.data = up_weight
    
    def project_down(self, latxa_embeds):
        """Project Latxa embeddings (8192) to Llama 3 size (4096)."""
        return self.down_projection(latxa_embeds)
    
    def project_up(self, llama3_embeds):
        """Project Llama 3 embeddings (4096) to Latxa size (8192)."""
        return self.up_projection(llama3_embeds)
    
    def save(self, path):
        """Save adapter weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Adapter saved to {path}")
    
    def load(self, path):
        """Load adapter weights."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"Adapter loaded from {path}")


# ==================== DYNAMIC AUGMENTER ====================
class DynamicAugmenter:
    """
    Runtime augmenter that:
      - takes dynamic tokens (strings) produced per-batch,
      - maps tokens already in latxa_vocab -> keep their ids,
      - for new tokens: allocate new ids, predict embeddings with hypernet,
        and write those embeddings into model's embedding matrix.
    """

    def __init__(self, model, latxa_tokenizer, hypernet, hypernet_tokenizer,
                 cache_limit=50000, device=None, adapter_path=None):
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
        
        # ==================== INITIALIZE ADAPTER ====================
        print("Initializing Latxa-to-Llama3 projection adapter...")
        
        # Get actual hidden sizes from the models
        latxa_hidden_size = model.config.hidden_size  # Should be 8192 for Latxa
        # Llama 3 8B has hidden_size 4096
        llama3_hidden_size = 4096
        
        print(f"Latxa hidden size: {latxa_hidden_size}")
        print(f"Llama 3 hidden size: {llama3_hidden_size}")
        
        self.adapter = LatxaToLlama3Adapter(
            latxa_hidden_size=latxa_hidden_size,
            llama3_hidden_size=llama3_hidden_size
        ).to(self.device)
        
        # Try to load pretrained adapter if provided or exists
        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading pretrained adapter from {adapter_path}")
            self.adapter.load(adapter_path)
        else:
            default_path = "models/latxa_llama3_adapter.pt"
            if os.path.exists(default_path):
                print(f"Loading pretrained adapter from {default_path}")
                self.adapter.load(default_path)
            else:
                print("No pretrained adapter found, using initialized projections")
        
        self.adapter.eval()  # Set to eval mode

    def _ensure_capacity(self, n_new):
        """Resize model embeddings to accomodate n_new new ids."""
        new_size = self.current_vocab_size + n_new
        if new_size > self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(new_size)
        self.current_vocab_size = self.model.get_input_embeddings().num_embeddings

    def _predict_embeddings_for_tokens(self, tokens_list):
        """
        Predict embeddings for dynamic tokens using Zett hypernetwork.
        Uses projection adapter to handle dimension mismatch.
        Returns dict[token] -> (in_emb, out_emb) on CPU.
        """

        # Convert tokens to char strings
        char_tokens = expand_to_char_tokens(tokens_list)
        char_strings = ["".join(chars) for chars in char_tokens]

        # Build surface form matrix
        surfaces = get_surface_form_matrix(
            char_strings,
            maxlen=self.hypernet.config.hn_surface_maxlen,
            tokenizer_to_use=self.hypernet_tokenizer
        )

        # Unpack tuple if needed
        if isinstance(surfaces, tuple):
            surfaces = surfaces[0]

        # Convert to tensor on device
        surfaces = torch.tensor(surfaces, dtype=torch.long, device=self.device)

        # Get source embeddings from Latxa (8192 dims)
        source_embeddings = self.model.get_input_embeddings().weight

        with torch.no_grad():
            # ==================== PROJECT DOWN ====================
            # Project source embeddings from 8192 -> 4096 for hypernet
            # source_embeddings shape: [vocab_size, 8192]
            source_embeddings_projected = self.adapter.project_down(source_embeddings)
            # source_embeddings_projected shape: [vocab_size, 4096]
            
            # ==================== RUN HYPERNET ====================
            # Now hypernet receives 4096-dim embeddings (correct size)
            pred_in_llama3, pred_out_llama3, _ = self.hypernet(
                surfaces,
                source_embeddings=source_embeddings_projected
            )
            # pred_in_llama3 shape: [batch, 4096]
            # pred_out_llama3 shape: [batch, 4096]
            
            # ==================== PROJECT UP ====================
            # Project predictions back to Latxa size: 4096 -> 8192
            pred_in = self.adapter.project_up(pred_in_llama3)
            pred_out = self.adapter.project_up(pred_out_llama3)
            # pred_in shape: [batch, 8192]
            # pred_out shape: [batch, 8192]

        # Return CPU tensors
        result = {}
        for i, tok in enumerate(tokens_list):
            result[tok] = (
                pred_in[i].detach().cpu(),
                pred_out[i].detach().cpu(),
            )

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
    
    def save_adapter(self, path="models/latxa_llama3_adapter.pt"):
        """Save the adapter for reuse."""
        self.adapter.save(path)


# ==================== HELPER FUNCTIONS ====================
def normalize_dynamic_token(tok: str) -> str:
    tok = tok.replace("Ġ", "").replace("▁", "").replace("<s>", "").replace("</s>", "").strip()
    if tok == "":
        tok = " "
    return tok

def expand_to_char_tokens(tokens):
    return [list(normalize_dynamic_token(t)) for t in tokens]


# ==================== OPTIONAL: TRAIN ADAPTER ====================
def train_adapter(augmenter, num_tokens=1000, learning_rate=1e-4, num_epochs=10):
    """
    Optional: Train the adapter to better align Latxa and Llama 3 embedding spaces.
    
    This trains the projection layers by:
    1. Taking existing Latxa token embeddings
    2. Projecting them through adapter + hypernet
    3. Comparing predicted embeddings with actual Latxa embeddings
    4. Minimizing the reconstruction error
    """
    import random
    from tqdm import tqdm
    
    print(f"\nTraining adapter to align embedding spaces...")
    print(f"Using {num_tokens} random tokens for {num_epochs} epochs")
    
    augmenter.adapter.train()
    optimizer = torch.optim.Adam(augmenter.adapter.parameters(), lr=learning_rate)
    
    # Get Latxa embeddings
    latxa_in_embeds = augmenter.model.get_input_embeddings().weight.data
    latxa_out_embeds = augmenter.model.get_output_embeddings().weight.data
    
    # Sample random tokens
    vocab_size = latxa_in_embeds.shape[0]
    token_indices = random.sample(range(min(vocab_size, augmenter.base_vocab_size)), 
                                  min(num_tokens, vocab_size))
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for idx in tqdm(token_indices, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            # Get actual embeddings
            true_in_embed = latxa_in_embeds[idx].unsqueeze(0)  # [1, 8192]
            true_out_embed = latxa_out_embeds[idx].unsqueeze(0)  # [1, 8192]
            
            # Project down
            projected = augmenter.adapter.project_down(true_in_embed)
            
            # Simulate what hypernet would do (just identity for training)
            # In real usage, hypernet transforms these, but for adapter training
            # we just want to minimize round-trip error
            pred_llama3 = projected  # [1, 4096]
            
            # Project back up
            pred_in = augmenter.adapter.project_up(pred_llama3)
            pred_out = augmenter.adapter.project_up(projected)
            
            # Compute loss (MSE between predicted and true embeddings)
            loss_in = nn.functional.mse_loss(pred_in, true_in_embed)
            loss_out = nn.functional.mse_loss(pred_out, true_out_embed)
            loss = loss_in + loss_out
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(token_indices)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.6f}")
    
    augmenter.adapter.eval()
    print("✓ Adapter training complete!")
    
    # Save trained adapter
    augmenter.save_adapter()
    
    return augmenter