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
    Adapter to bridge Latxa (4096 hidden dim) to Llama 3 hypernet (8192 hidden dim).
    """
    def __init__(self, latxa_hidden_size=4096, llama3_hidden_size=8192):
        super().__init__()
        self.latxa_hidden_size = latxa_hidden_size
        self.llama3_hidden_size = llama3_hidden_size
        
        # Project Latxa embeddings UP to Llama 3 size (for hypernet input)
        self.up_projection = nn.Linear(latxa_hidden_size, llama3_hidden_size, bias=False)
        
        # Project Llama 3 embeddings DOWN to Latxa size (for hypernet output)
        self.down_projection = nn.Linear(llama3_hidden_size, latxa_hidden_size, bias=False)
        
        # Initialize projections
        self._initialize_projections()
        
        # Initialize projections
        self._initialize_projections()
    
    def _initialize_projections(self):
        """Initialize projections intelligently."""
        with torch.no_grad():
            if self.latxa_hidden_size == self.llama3_hidden_size:
                # Same size - just use identity
                self.up_projection.weight.data = torch.eye(self.llama3_hidden_size)
                self.down_projection.weight.data = torch.eye(self.latxa_hidden_size)
            elif self.latxa_hidden_size < self.llama3_hidden_size:
                # Up projection: repeat/pad Latxa (4096) -> Llama3 (8192)
                up_weight = torch.zeros(self.llama3_hidden_size, self.latxa_hidden_size)
                # Copy first half
                up_weight[:self.latxa_hidden_size, :] = torch.eye(self.latxa_hidden_size)
                # Copy to second half
                up_weight[self.latxa_hidden_size:, :] = torch.eye(self.latxa_hidden_size)
                self.up_projection.weight.data = up_weight
                
                # Down projection: average Llama3 (8192) -> Latxa (4096)
                down_weight = torch.zeros(self.latxa_hidden_size, self.llama3_hidden_size)
                for i in range(self.latxa_hidden_size):
                    # Average two dimensions
                    down_weight[i, 2*i] = 0.5
                    down_weight[i, 2*i + 1] = 0.5
                self.down_projection.weight.data = down_weight
            else:
                # latxa_hidden_size > llama3_hidden_size (shouldn't happen now)
                # Down projection: average dimensions
                down_weight = torch.zeros(self.llama3_hidden_size, self.latxa_hidden_size)
                ratio = self.latxa_hidden_size / self.llama3_hidden_size
                for i in range(self.llama3_hidden_size):
                    start_idx = int(i * ratio)
                    end_idx = int((i + 1) * ratio)
                    for j in range(start_idx, end_idx):
                        down_weight[i, j] = 1.0 / (end_idx - start_idx)
                self.down_projection.weight.data = down_weight
                
                # Up projection: copy and repeat
                up_weight = torch.zeros(self.latxa_hidden_size, self.llama3_hidden_size)
                for i in range(self.latxa_hidden_size):
                    src_idx = int(i * self.llama3_hidden_size / self.latxa_hidden_size)
                    up_weight[i, src_idx] = 1.0
                self.up_projection.weight.data = up_weight
    
    def project_to_llama3(self, latxa_embeds):
        """Project Latxa embeddings (4096) to Llama 3 size (8192)."""
        return self.up_projection(latxa_embeds)
    
    def project_to_latxa(self, llama3_embeds):
        """Project Llama 3 embeddings (8192) to Latxa size (4096)."""
        return self.down_projection(llama3_embeds)
    
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
        
        # Get actual hidden sizes from the embeddings
        actual_embed_size = model.get_input_embeddings().weight.shape[1]
        latxa_hidden_size = actual_embed_size  # Latxa: 4096
        
        # Check hypernet's actual output size by looking at its config
        if hasattr(hypernet.config, 'hn_model_size'):
            llama3_hidden_size = hypernet.config.hn_model_size
        elif hasattr(hypernet.config, 'hidden_size'):
            llama3_hidden_size = hypernet.config.hidden_size
        else:
            # Default guess - but let's verify after first call
            llama3_hidden_size = 4096
            print("WARNING: Could not determine hypernet output size, assuming 4096")
        
        print(f"Latxa actual embedding size: {latxa_hidden_size}")
        print(f"Latxa config hidden size: {model.config.hidden_size}")
        print(f"Llama 3 hypernet expected size: {llama3_hidden_size}")
        
        # If both are the same size, we don't need projection
        if latxa_hidden_size == llama3_hidden_size:
            print("Both models have same hidden size - adapter will use identity")
        
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
        Uses Latxa's embeddings (projected) as source context for the hypernet.
        Returns dict[token] -> (in_emb, out_emb) on CPU.
        """

        # Convert tokens to char strings
        char_tokens = expand_to_char_tokens(tokens_list)
        char_strings = ["".join(chars) for chars in char_tokens]

        # Build surface form matrix using hypernet's tokenizer
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

        # ==================== BUILD ALIGNED SOURCE EMBEDDINGS ====================
        # Get Latxa embeddings
        latxa_embeddings = self.model.get_input_embeddings().weight  # [32000, 4096]
        
        # Get hypernet vocab size
        hypernet_vocab_size = self.hypernet_tokenizer.vocab_size
        
        # Get the actual output size from adapter (in case they're the same)
        target_dim = self.adapter.llama3_hidden_size
        
        # Create aligned embedding matrix for hypernet's vocab
        aligned_embeddings = torch.zeros(
            hypernet_vocab_size, 
            target_dim,
            dtype=latxa_embeddings.dtype,
            device=self.device
        )
        
        # Map overlapping tokens from Latxa to Llama3 vocab
        for token, latxa_id in self.vocab.items():
            if token in self.hypernet_tokenizer.get_vocab():
                llama3_id = self.hypernet_tokenizer.get_vocab()[token]
                if llama3_id < hypernet_vocab_size:
                    latxa_emb = latxa_embeddings[latxa_id]
                    # Project if needed (will be identity if same size)
                    projected_emb = self.adapter.project_to_llama3(latxa_emb.unsqueeze(0)).squeeze(0)
                    aligned_embeddings[llama3_id] = projected_emb
        
        # For unmapped tokens, use small random values
        unmapped_mask = (aligned_embeddings.sum(dim=1) == 0)
        if unmapped_mask.any():
            aligned_embeddings[unmapped_mask] = torch.randn(
                unmapped_mask.sum(), target_dim,
                dtype=aligned_embeddings.dtype,
                device=self.device
            ) * 0.02

        with torch.no_grad():
            try:
                pred_in_llama3, pred_out_llama3, _ = self.hypernet(
                    surfaces,
                    source_embeddings=aligned_embeddings
                )
            except RuntimeError as e:
                print(f"ERROR in hypernet call:")
                print(f"  surfaces shape: {surfaces.shape}")
                print(f"  aligned_embeddings shape: {aligned_embeddings.shape}")
                print(f"  surfaces min/max: {surfaces.min()}/{surfaces.max()}")
                raise e
            
            # Project predictions to Latxa size (will be identity if same size)
            pred_in = self.adapter.project_to_latxa(pred_in_llama3)
            pred_out = self.adapter.project_to_latxa(pred_out_llama3)

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