"""
Emformer Layer Implementation for EchoStream

Based on:
- "Emformer: Efficient Memory Transformer Based Acoustic Model For Low Latency Streaming Speech Recognition"
- Paper: https://arxiv.org/abs/2010.10759

Key innovations:
1. Left Context Cache: Reuse K, V from previous segments (efficiency)
2. Memory Bank from lower layer: Enable parallelized training
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import math


class EmformerEncoderLayer(nn.Module):
    """
    Single Emformer Encoder Layer.
    
    Key differences from AM-TRF (Attention Masking Transformer):
    1. Left Context K, V are cached from previous segments (not recomputed)
    2. Memory Bank M comes from lower layer (n-1), not same layer
    
    This enables:
    - Reduced computation (no redundant K, V calculation for left context)
    - Parallelized training (memory bank flows across layers)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        segment_length: int = 4,
        left_context_length: int = 30,
        right_context_length: int = 0,
        memory_size: int = 8,
        ffn_embed_dim: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Architecture parameters
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.memory_size = memory_size
        
        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=False,  # [T, B, D] format
        )
        
        # Feed-Forward Network (FFN)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        
        # Layer Normalization (Pre-LN style)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(activation_dropout)
        
        # Activation
        self.activation_fn = nn.ReLU()
        
        # Memory Bank projection (optional)
        # For summarization: mean pooling of [M, L, C, R]
        self.memory_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        center: torch.Tensor,
        right: Optional[torch.Tensor] = None,
        left_context_key: Optional[torch.Tensor] = None,
        left_context_value: Optional[torch.Tensor] = None,
        memory_bank: Optional[torch.Tensor] = None,
        summary_query: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of Emformer layer.
        
        Args:
            center: Current center segment [T_c, B, D]
            right: Right context (lookahead) [T_r, B, D], optional
            left_context_key: Cached K from previous segments [T_l, B, D]
            left_context_value: Cached V from previous segments [T_l, B, D]
            memory_bank: Memory from lower layer (n-1) [M, B, D]
            summary_query: Query for summary generation [1, B, D], optional
        
        Returns:
            center_out: Processed center segment [T_c, B, D]
            right_out: Processed right context [T_r, B, D] or None
            cache: Dict containing:
                - 'key': K for next segment's left context
                - 'value': V for next segment's left context
                - 'memory': Memory bank for upper layer (n+1)
        """
        # ========================================
        # Step 1: Prepare Query, Key, Value
        # ========================================
        
        # Query: C, R, S (center, right, summary)
        # Note: Left context does NOT generate Q (efficiency!)
        queries = []
        
        # Center query
        q_center = center
        queries.append(q_center)
        
        # Right query (if lookahead enabled)
        q_right = None
        if right is not None:
            q_right = right
            queries.append(q_right)
        
        # Summary query (for memory bank generation)
        if summary_query is not None:
            queries.append(summary_query)
        
        query = torch.cat(queries, dim=0)  # [T_c + T_r + 1, B, D]
        
        # Key, Value: M, L, C, R
        keys = []
        values = []
        
        # Memory bank K, V
        if memory_bank is not None:
            keys.append(memory_bank)
            values.append(memory_bank)
        
        # Left context K, V (CACHED from previous segments)
        # This is the key efficiency improvement!
        if left_context_key is not None and left_context_value is not None:
            keys.append(left_context_key)
            values.append(left_context_value)
        
        # Center K, V (computed fresh)
        keys.append(center)
        values.append(center)
        
        # Right K, V (computed fresh)
        if right is not None:
            keys.append(right)
            values.append(right)
        
        key = torch.cat(keys, dim=0)  # [M + T_l + T_c + T_r, B, D]
        value = torch.cat(values, dim=0)
        
        # ========================================
        # Step 2: Self-Attention
        # ========================================
        
        residual = query
        query = self.self_attn_layer_norm(query)
        
        # Multi-head attention
        attn_output, attn_weights = self.self_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        
        query = residual + self.dropout(attn_output)
        
        # ========================================
        # Step 3: Feed-Forward Network
        # ========================================
        
        residual = query
        query = self.final_layer_norm(query)
        
        query = self.fc1(query)
        query = self.activation_fn(query)
        query = self.activation_dropout(query)
        query = self.fc2(query)
        query = self.dropout(query)
        
        output = residual + query  # [T_c + T_r + 1, B, D]
        
        # ========================================
        # Step 4: Split output and prepare cache
        # ========================================
        
        # Split output back to center, right, summary
        T_c = center.size(0)
        T_r = right.size(0) if right is not None else 0
        
        center_out = output[:T_c]  # [T_c, B, D]
        
        right_out = None
        if T_r > 0:
            right_out = output[T_c:T_c + T_r]  # [T_r, B, D]
        
        # Summary output (for memory bank to upper layer)
        summary_out = None
        if summary_query is not None:
            summary_out = output[-1:]  # [1, B, D]
        
        # ========================================
        # Step 5: Generate cache for next segment
        # ========================================
        
        # Cache current center's K, V for next segment's left context
        # In next segment, this will be used as left_context_key/value
        cache_key = center  # Current center becomes left context
        cache_value = center
        
        # Generate memory bank for upper layer (n+1)
        # Memory: mean of [M, L, C, R] or use summary
        if summary_out is not None:
            memory_out = self.memory_proj(summary_out)
        else:
            # Alternative: mean pooling
            all_context = torch.cat([
                memory_bank if memory_bank is not None else torch.zeros(0, center.size(1), center.size(2), device=center.device),
                left_context_key if left_context_key is not None else torch.zeros(0, center.size(1), center.size(2), device=center.device),
                center,
                right if right is not None else torch.zeros(0, center.size(1), center.size(2), device=center.device),
            ], dim=0)
            memory_out = all_context.mean(dim=0, keepdim=True)  # [1, B, D]
            memory_out = self.memory_proj(memory_out)
        
        cache = {
            'key': cache_key,
            'value': cache_value,
            'memory': memory_out,
        }
        
        return center_out, right_out, cache


class EmformerEncoder(nn.Module):
    """
    Multi-layer Emformer Encoder.
    
    Processes input in segments with:
    1. Left context cache (horizontal reuse across segments)
    2. Memory bank flow (vertical flow across layers)
    """
    
    def __init__(
        self,
        num_layers: int = 16,
        embed_dim: int = 256,
        num_heads: int = 4,
        segment_length: int = 4,
        left_context_length: int = 30,
        right_context_length: int = 0,
        memory_size: int = 8,
        ffn_embed_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.segment_length = segment_length
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.memory_size = memory_size
        
        # Emformer layers
        self.layers = nn.ModuleList([
            EmformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_length=segment_length,
                left_context_length=left_context_length,
                right_context_length=right_context_length,
                memory_size=memory_size,
                ffn_embed_dim=ffn_embed_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Cache storage
        self.reset_cache()
    
    def reset_cache(self):
        """Reset all caches for new utterance."""
        # Left context cache: stores K, V for each layer
        # Shape: list of dicts per layer
        self.left_context_cache = [
            {'key': [], 'value': []}
            for _ in range(self.num_layers)
        ]
        
        # Memory bank: stores memory for each layer
        # M_i^n comes from layer (n-1), segment (i-1)
        self.memory_bank = [None for _ in range(self.num_layers)]
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with segment-wise processing.
        
        Args:
            x: Input features [T, B, D]
            lengths: Sequence lengths [B]
        
        Returns:
            Dict with:
                - 'encoder_out': Output features [T, B, D]
                - 'encoder_padding_mask': Padding mask [B, T]
        """
        T, B, D = x.size()
        
        # Segment the input
        S = self.segment_length
        R = self.right_context_length
        num_segments = (T + S - 1) // S
        
        outputs = []
        
        # Process each segment
        for seg_idx in range(num_segments):
            # Get center segment
            center_start = seg_idx * S
            center_end = min(center_start + S, T)
            center = x[center_start:center_end]  # [S, B, D]
            
            # Get right context (lookahead)
            right = None
            if R > 0 and center_end < T:
                right_end = min(center_end + R, T)
                right = x[center_end:right_end]  # [R, B, D]
            
            # Process through layers
            layer_input_center = center
            layer_input_right = right
            
            for layer_idx, layer in enumerate(self.layers):
                # Get left context cache from previous segments
                left_key = None
                left_value = None
                if len(self.left_context_cache[layer_idx]['key']) > 0:
                    # Use last L segments as left context
                    L = min(self.left_context_length // S, len(self.left_context_cache[layer_idx]['key']))
                    left_key = torch.cat(self.left_context_cache[layer_idx]['key'][-L:], dim=0)
                    left_value = torch.cat(self.left_context_cache[layer_idx]['value'][-L:], dim=0)
                
                # Get memory bank from lower layer (n-1)
                # For layer 0, memory is None (no lower layer)
                memory = None
                if layer_idx > 0:
                    memory = self.memory_bank[layer_idx - 1]
                
                # Forward through layer
                layer_output_center, layer_output_right, cache = layer(
                    center=layer_input_center,
                    right=layer_input_right,
                    left_context_key=left_key,
                    left_context_value=left_value,
                    memory_bank=memory,
                )
                
                # Update cache
                # Cache current center's K, V for next segment
                self.left_context_cache[layer_idx]['key'].append(cache['key'])
                self.left_context_cache[layer_idx]['value'].append(cache['value'])
                
                # Limit cache size
                max_cache_segments = (self.left_context_length // S) + 2
                if len(self.left_context_cache[layer_idx]['key']) > max_cache_segments:
                    self.left_context_cache[layer_idx]['key'].pop(0)
                    self.left_context_cache[layer_idx]['value'].pop(0)
                
                # Update memory bank for upper layer (n+1)
                self.memory_bank[layer_idx] = cache['memory']
                
                # Prepare input for next layer
                layer_input_center = layer_output_center
                layer_input_right = layer_output_right
            
            # Collect output
            outputs.append(layer_input_center)
            if layer_input_right is not None and seg_idx == num_segments - 1:
                # Include right context in last segment
                outputs.append(layer_input_right)
        
        # Concatenate all segments
        encoder_out = torch.cat(outputs, dim=0)  # [T, B, D]
        
        # Generate padding mask if lengths provided
        encoder_padding_mask = None
        if lengths is not None:
            max_len = encoder_out.size(0)
            encoder_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        return {
            'encoder_out': [encoder_out],
            'encoder_padding_mask': [encoder_padding_mask] if encoder_padding_mask is not None else [],
        }


if __name__ == "__main__":
    # Test Emformer Layer
    print("Testing EmformerEncoderLayer...")
    layer = EmformerEncoderLayer(
        embed_dim=256,
        num_heads=4,
        segment_length=4,
        left_context_length=30,
        right_context_length=0,
        memory_size=8,
    )
    
    # Test inputs
    center = torch.randn(4, 2, 256)  # [T_c=4, B=2, D=256]
    right = None  # No lookahead
    left_key = torch.randn(30, 2, 256)  # [T_l=30, B=2, D=256]
    left_value = torch.randn(30, 2, 256)
    memory = torch.randn(8, 2, 256)  # [M=8, B=2, D=256]
    
    center_out, right_out, cache = layer(
        center=center,
        right=right,
        left_context_key=left_key,
        left_context_value=left_value,
        memory_bank=memory,
    )
    
    print(f"Center output shape: {center_out.shape}")
    print(f"Cache key shape: {cache['key'].shape}")
    print(f"Cache value shape: {cache['value'].shape}")
    print(f"Cache memory shape: {cache['memory'].shape}")
    
    # Test Emformer Encoder
    print("\nTesting EmformerEncoder...")
    encoder = EmformerEncoder(
        num_layers=4,
        embed_dim=256,
        num_heads=4,
        segment_length=4,
        left_context_length=30,
    )
    
    # Test input
    x = torch.randn(100, 2, 256)  # [T=100, B=2, D=256]
    lengths = torch.tensor([100, 80])
    
    output = encoder(x, lengths)
    print(f"Encoder output shape: {output['encoder_out'][0].shape}")
    print(f"Padding mask shape: {output['encoder_padding_mask'][0].shape if output['encoder_padding_mask'] else None}")
    
    print("\n✅ Emformer implementation test passed!")

