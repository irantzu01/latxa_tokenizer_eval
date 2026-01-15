#!/usr/bin/env python3
"""
FOCUS: Effective Embedding Initialization for Tokenizer Transfer
Combines multiple initialization strategies for optimal results.

Based on: "FOCUS: Effective Embedding Initialization for Monolingual 
Specialization of Multilingual Models" (2023)
"""

import torch
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np

class FOCUSInitializer:
    """
    Implements FOCUS initialization combining:
    1. Exact match copying
    2. Subword composition (for BPE tokens)
    3. String similarity matching
    4. Mean + noise fallback
    """
    
    def __init__(self, old_tokenizer, new_tokenizer, old_embeddings, device='cuda'):
        """
        Args:
            old_tokenizer: Original tokenizer (e.g., Latxa)
            new_tokenizer: New tokenizer (e.g., Basque)
            old_embeddings: Existing embedding matrix [vocab_size, hidden_dim]
            device: torch device
        """
        self.old_tokenizer = old_tokenizer
        self.new_tokenizer = new_tokenizer
        self.old_embeddings = old_embeddings
        self.device = device
        
        self.old_vocab = old_tokenizer.get_vocab()
        self.new_vocab = new_tokenizer.get_vocab()
        self.reverse_new_vocab = {v: k for k, v in self.new_vocab.items()}
        
        # Statistics
        self.stats = {
            'exact_match': 0,
            'subword_composition': 0,
            'similarity_match': 0,
            'fallback': 0
        }
    
    def initialize_embeddings(self):
        """
        Initialize new embedding matrix using FOCUS strategy.
        
        Returns:
            new_embeddings: Initialized embedding tensor [new_vocab_size, hidden_dim]
            old_token_ids: List of token IDs that existed in old vocab
            new_token_ids: List of token IDs that are new
        """
        print("\n" + "="*60)
        print("FOCUS Embedding Initialization")
        print("="*60)
        
        vocab_size = len(self.new_vocab)
        hidden_dim = self.old_embeddings.shape[1]
        
        # Create new embedding matrix
        new_embeddings = torch.zeros(
            vocab_size, 
            hidden_dim,
            dtype=self.old_embeddings.dtype,
            device=self.device
        )
        
        old_token_ids = []
        new_token_ids = []
        
        print(f"Processing {vocab_size:,} tokens...")
        
        for new_idx in range(vocab_size):
            new_token = self.reverse_new_vocab[new_idx]
            
            # Strategy 1: Exact match (copy directly)
            if new_token in self.old_vocab:
                old_idx = self.old_vocab[new_token]
                new_embeddings[new_idx] = self.old_embeddings[old_idx]
                old_token_ids.append(new_idx)
                self.stats['exact_match'] += 1
                continue
            
            # This is a new token - try initialization strategies
            new_token_ids.append(new_idx)
            
            # Strategy 2: Subword composition
            composed_emb = self._subword_composition(new_token)
            if composed_emb is not None:
                new_embeddings[new_idx] = composed_emb
                self.stats['subword_composition'] += 1
                continue
            
            # Strategy 3: String similarity
            similar_emb = self._similarity_matching(new_token, top_k=5)
            if similar_emb is not None:
                new_embeddings[new_idx] = similar_emb
                self.stats['similarity_match'] += 1
                continue
            
            # Strategy 4: Fallback (mean + noise)
            new_embeddings[new_idx] = self._fallback_initialization()
            self.stats['fallback'] += 1
        
        # Print statistics
        print(f"\nInitialization Statistics:")
        print(f"  Exact matches: {self.stats['exact_match']:,} ({100*self.stats['exact_match']/vocab_size:.1f}%)")
        print(f"  Subword composition: {self.stats['subword_composition']:,} ({100*self.stats['subword_composition']/vocab_size:.1f}%)")
        print(f"  Similarity matching: {self.stats['similarity_match']:,} ({100*self.stats['similarity_match']/vocab_size:.1f}%)")
        print(f"  Fallback (mean+noise): {self.stats['fallback']:,} ({100*self.stats['fallback']/vocab_size:.1f}%)")
        print(f"\nTotal new tokens: {len(new_token_ids):,}")
        print(f"Total old tokens: {len(old_token_ids):,}")
        
        return new_embeddings, old_token_ids, new_token_ids
    
    def _subword_composition(self, new_token):
        """
        Initialize by averaging embeddings of constituent subwords.
        Works well for BPE tokens that are compositions of existing pieces.
        """
        # Remove special characters for tokenization
        clean_token = new_token.replace('▁', ' ').replace('Ġ', ' ').strip()
        
        if not clean_token:
            return None
        
        # Tokenize with old tokenizer to get subwords
        try:
            subword_tokens = self.old_tokenizer.tokenize(clean_token)
        except:
            return None
        
        if not subword_tokens or len(subword_tokens) > 10:  # Avoid too many subwords
            return None
        
        # Collect embeddings of subwords
        subword_embeds = []
        for sw in subword_tokens:
            if sw in self.old_vocab:
                old_idx = self.old_vocab[sw]
                subword_embeds.append(self.old_embeddings[old_idx])
        
        if len(subword_embeds) >= 1:
            # Weighted average: longer subwords get more weight
            if len(subword_embeds) == 1:
                return subword_embeds[0]
            else:
                # Simple average (FOCUS paper shows this works best)
                return torch.stack(subword_embeds).mean(dim=0)
        
        return None
    
    def _similarity_matching(self, new_token, top_k=5):
        """
        Find similar tokens by string similarity and average their embeddings.
        Uses character-level edit distance.
        """
        # Clean token for comparison
        clean_new = new_token.replace('▁', '').replace('Ġ', '').strip()
        
        if len(clean_new) < 2:  # Too short for meaningful similarity
            return None
        
        # Find similar tokens
        similarities = []
        for old_token in self.old_vocab.keys():
            clean_old = old_token.replace('▁', '').replace('Ġ', '').strip()
            
            if len(clean_old) < 2:
                continue
            
            # Calculate similarity ratio
            sim = SequenceMatcher(None, clean_new, clean_old).ratio()
            
            # Only consider reasonably similar tokens (>0.5 similarity)
            if sim > 0.5:
                similarities.append((old_token, sim))
        
        if not similarities:
            return None
        
        # Sort by similarity and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]
        
        # Get embeddings
        similar_embeds = []
        weights = []
        for old_token, sim in top_similar:
            old_idx = self.old_vocab[old_token]
            similar_embeds.append(self.old_embeddings[old_idx])
            weights.append(sim)
        
        # Weighted average based on similarity scores
        weights_tensor = torch.tensor(weights, dtype=self.old_embeddings.dtype, device=self.device)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize
        
        weighted_emb = torch.zeros_like(similar_embeds[0])
        for emb, weight in zip(similar_embeds, weights_tensor):
            weighted_emb += emb * weight
        
        return weighted_emb
    
    def _fallback_initialization(self):
        """
        Fallback: initialize with mean of existing embeddings + small noise.
        """
        mean_emb = self.old_embeddings.mean(dim=0)
        std_emb = self.old_embeddings.std(dim=0).mean().item()
        
        noise = torch.randn(
            self.old_embeddings.shape[1],
            dtype=self.old_embeddings.dtype,
            device=self.device
        ) * (std_emb * 0.1)
        
        return mean_emb + noise


# ================= Function to call in training script ===================
def initialize_embeddings_with_focus(model, old_tokenizer, new_tokenizer, device):
    """
    Initialize model embeddings using FOCUS strategy.
    
    Args:
        model: The Latxa model
        old_tokenizer: Original Latxa tokenizer
        new_tokenizer: New Basque tokenizer
        device: torch device
    
    Returns:
        old_token_ids: List of token IDs from old vocab
        new_token_ids: List of new token IDs
    """
    print("\n" + "="*60)
    print("Step 4: Embedding Alignment with FOCUS")
    print("="*60)
    
    # Get current embeddings
    old_embeddings = model.get_input_embeddings().weight.data
    
    # Initialize with FOCUS
    initializer = FOCUSInitializer(
        old_tokenizer=old_tokenizer,
        new_tokenizer=new_tokenizer,
        old_embeddings=old_embeddings,
        device=device
    )
    
    new_embeddings, old_token_ids, new_token_ids = initializer.initialize_embeddings()
    
    # Resize model embeddings
    model.resize_token_embeddings(len(new_tokenizer))
    
    # Set new embeddings
    model.get_input_embeddings().weight.data = new_embeddings
    
    # Also initialize output embeddings (LM head)
    # Use same initialization for consistency
    output_embeddings = model.get_output_embeddings().weight.data
    model.get_output_embeddings().weight.data = new_embeddings.clone()
    
    print(f"\n✓ Embeddings initialized with FOCUS")
    print(f"  Vocab size: {len(new_tokenizer):,}")
    print(f"  Old tokens: {len(old_token_ids):,}")
    print(f"  New tokens: {len(new_token_ids):,}")
    
    return old_token_ids, new_token_ids
